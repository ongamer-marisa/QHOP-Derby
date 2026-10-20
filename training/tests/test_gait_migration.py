import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from qhop_train.ppo import PPO
from qhop_train.schema import LEGACY_OBSERVATION, OBSERVATION, VERSION
from qhop_train.migration import adapt, R11_REWARD, R11_COURSE
from qhop_train import checkpoint

class GaitMigrationTests(unittest.TestCase):
    def test_old_migration_rejected_without_touching_source_or_destination(self):
        torch.set_num_threads(2)
        p=PPO("cpu")
        original={k:v.clone() for k,v in p.actor.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"old.pt";path.write_bytes(b"old unchanged checkpoint")
            with self.assertRaisesRegex(ValueError,"from scratch"):adapt(path,p)
            self.assertEqual(path.read_bytes(),b"old unchanged checkpoint")
            self.assertTrue(all(torch.equal(v,p.actor.state_dict()[k]) for k,v in original.items()))

    def test_pilot_gate_requires_all_diagnostics(self):
        from qhop_train.curriculum import gait_pilot_passes
        report=dict(episodes=100,diagnostic_episodes=100,correct_cycle_ratio=.2,
            fall_free=.85,four_leg_participation=.9,synchrony_cycle_ratio=.2,speed_band_fraction=.8)
        self.assertTrue(gait_pilot_passes(report))
        for key in report:
            self.assertFalse(gait_pilot_passes({k:v for k,v in report.items() if k!=key}))

    def test_batched_bootstrap_keeps_terminals_and_streams(self):
        from qhop_train.runner import collect
        from unittest.mock import Mock
        ppo=PPO("cpu")
        class Value(torch.nn.Module):
            def __init__(self):super().__init__();self.batches=[]
            def forward(self,obs,visual=None):self.batches.append(len(obs));return obs[:,0]*2
        ppo.critic=Value()
        raw=np.zeros((2,OBSERVATION),dtype=np.float32);raw[:,0]=[1,2]
        initial=dict(decision_id=np.array([1,2]),decision_obs=raw,
            terminal_id=np.array([],dtype=int))
        next_raw=raw.copy();next_raw[:,0]=[3,4]
        following=dict(decision_id=np.array([2]),decision_obs=next_raw[:1],decision_reward=np.array([20.]),
            terminal_id=np.array([1]),terminal_obs=next_raw[1:],terminal_reward=np.array([10.]))
        frozen=dict(decision_id=np.array([],dtype=int),decision_obs=np.zeros((0,OBSERVATION)),terminal_id=np.array([],dtype=int))
        class Workers:
            current=[dict(learner=initial,frozen=frozen),dict(learner=initial,frozen=frozen)]
            def step(self,submitted):return [dict(learner=following,frozen=frozen)]*2
        opponents=Mock();opponents.actions.return_value=np.zeros((0,28),dtype=np.float32)
        rollout,_,_,_=collect(ppo,Workers(),opponents,target=4)
        np.testing.assert_array_equal(rollout.done[:,0],[True,False,True,False])
        np.testing.assert_allclose(rollout.next_value[:,0],[0,6,0,6],atol=1e-6)
        np.testing.assert_array_equal(rollout.reward[:,0],[10,20,10,20])
        np.testing.assert_array_equal(rollout.stream_id,[1,2,3,4])
        self.assertEqual(ppo.critic.batches,[2,2,2])
