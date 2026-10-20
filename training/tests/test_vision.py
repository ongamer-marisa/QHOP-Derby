import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from qhop_train.ppo import PPO,Rollout
from qhop_train.schema import *
from qhop_train.export import export,DeploymentActor
from qhop_train.checkpoint import save,load


class VisionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(14)

    def test_event_freeze_peek_and_reset(self):
        p=PPO();x=np.zeros(OBSERVATION,np.float32);x[SLICES.scan_id]=1
        h,_,_,_=p.vision_states.consume('a',x)
        for _ in range(3):self.assertTrue(torch.equal(h,p.vision_states.consume('a',x)[0]))
        x[SLICES.scan_id]=2;x[SLICES.lidar]=1
        preview=p.vision_states.consume('a',x,commit=False)[0]
        self.assertEqual(p.vision_states.entry('a')['updates'],1)
        actual=p.vision_states.consume('a',x)[0];self.assertTrue(torch.equal(preview,actual))
        self.assertFalse(torch.equal(h,actual))
        other=p.vision_states.consume('b',x)[0];self.assertFalse(torch.equal(actual,other))
        p.vision_states.terminal('a');fresh=p.vision_states.consume('a',x)[0]
        self.assertTrue(torch.equal(fresh,other))

    def test_expanded_cell_matches_grucell_and_temporal_gradients(self):
        p=PPO();scan=torch.randn(3,VISION_FEATURES);h=torch.randn(3,VISION_HIDDEN)
        expected=p.vision.gru(p.vision.encoder(scan/p.vision.scale),h)
        self.assertLess((p.vision(scan,h)-expected).abs().max().item(),1e-6)
        first=torch.randn(1,VISION_FEATURES,requires_grad=True)
        h=p.vision(first,torch.zeros(1,128));h=p.vision(torch.randn(1,VISION_FEATURES),h)
        h.square().sum().backward();self.assertGreater(first.grad.abs().sum().item(),0)

    def test_ppo_updates_shared_visual_parameters_and_refreshes_memory(self):
        p=PPO(total_updates=10);p.schedule.update=1
        x=np.random.default_rng(1).normal(size=(16,OBSERVATION)).astype(np.float32)
        x[:,SLICES.scan_id]=np.arange(16)//4+1
        memories=[p.vision_states.consume('a',row) for row in x]
        visual=torch.cat([item[0] for item in memories])
        with torch.no_grad():a,lp=p.actor.sample(torch.tensor(x),visual);v=p.critic(torch.tensor(x),visual)
        r=Rollout(x[:,None],a.numpy()[:,None],lp.numpy()[:,None],v.numpy()[:,None],np.zeros((16,1),np.float32),
            np.arange(16,dtype=np.float32)[:,None]/16,np.zeros((16,1),bool),np.ones((16,1),np.float32),
            np.zeros(16,np.int64),np.array([i[1] for i in memories]),[i[2] for i in memories])
        before=[q.detach().clone() for q in p.vision.parameters()]
        p.update(r,epochs=1)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,p.vision.parameters())))
        h=torch.zeros(1,128)
        with torch.no_grad():
            for s in p.vision_states.entry('a')['scans']:h=p.vision(torch.tensor(s)[None],h)
        self.assertTrue(torch.equal(h,p.vision_states.entry('a')['hidden']))
        ids=[id(q) for group in p.optimizer.param_groups for q in group['params']]
        self.assertEqual(len(ids),len(set(ids)))

    def test_two_onnx_models_match_over_scan_sequence(self):
        import onnx,onnxruntime as ort
        p=PPO()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);save(path/'p.pt',p,{'stage':1},{'physics_fingerprint':PHYSICS_FINGERPRINT})
            export(path/'p.pt',path/'QHopPolicy.onnx')
            vision=onnx.load(str(path/'QHopPolicyVision.onnx'))
            self.assertFalse(any(node.op_type in ('GRU','Loop','If') for node in vision.graph.node))
            sessions=[ort.InferenceSession(str(path/name),providers=['CPUExecutionProvider']) for name in ('QHopPolicy.onnx','QHopPolicyVision.onnx')]
            h=torch.zeros(2,128);actual=np.zeros((2,128),np.float32)
            rng=np.random.default_rng(9)
            for step in range(16):
                x=torch.tensor(rng.normal(size=(2,OBSERVATION)).astype(np.float32))
                if step%4==0:
                    with torch.no_grad():h=p.vision(x[:,SLICES.lidar],h)
                    actual=sessions[1].run(None,{'scan':x[:,SLICES.lidar].numpy(),'hidden':actual})[0]
                with torch.no_grad():expected=DeploymentActor(p.actor,p.normalizer.mean,p.normalizer.var)(x,h).numpy()
                output=sessions[0].run(None,{'observation':x.numpy(),'hidden':actual})[0]
                self.assertLess(float(np.max(np.abs(h.numpy()-actual))),1e-4)
                self.assertLess(float(np.max(np.abs(output-expected))),1e-4)

    def test_old_checkpoint_rejected_before_loading_weights(self):
        p=PPO()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt';torch.save({'schema_version':3},path)
            with self.assertRaisesRegex(ValueError,'incompatible'):load(path,p)
