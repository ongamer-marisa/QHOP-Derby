import math
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from importlib.util import find_spec

import numpy as np

from qhop_train.curriculum import (PHASES, difficulty, shaping,
                                   stage6_can_advance, stage6_final_ready,
                                   STAGE1_TASKS, REWARD_HASH, COURSE_HASH,
                                   task, evaluation_interval, stage1_result)
from qhop_train.schema import ACTION, FRAME, HISTORY, OBSERVATION, PHYSICS_FINGERPRINT, normalization_mask
from qhop_train.control import (Session, StopRequested, approve, freeze_candidate,
                                record_preview, request_stop, review_state)


class SchemaTests(unittest.TestCase):
    def test_build_contract_rejects_stale_binary(self):
        from qhop_train.build_contract import verify
        with tempfile.TemporaryDirectory() as directory:
            player=Path(directory)/"QHopTraining.exe"
            player.write_bytes(b"test player")
            data=Path(directory)/"QHopTraining_Data"
            (data/"Managed").mkdir(parents=True)
            (data/"Managed/Assembly-CSharp.dll").write_bytes(b"test assembly")
            (data/"resources.assets").write_bytes(b"test resources")
            (data/"level0").write_bytes(b"test scene")
            files=(player,data/"Managed/Assembly-CSharp.dll",data/"resources.assets",
                   data/"level0")
            hashes=[hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
            player.with_suffix(".contract.json").write_text(json.dumps({
                "schema_version":2,"observation_size":OBSERVATION,
                "action_size":ACTION,"physics_fingerprint":PHYSICS_FINGERPRINT,
                "reward_hash":REWARD_HASH,"course_hash":COURSE_HASH,
                **dict(zip(("executable_sha256","assembly_sha256","resources_sha256",
                            "scene_sha256"),hashes))}))
            self.assertEqual(verify(player),hashlib.sha256("".join(hashes).encode()).hexdigest())
            player.write_bytes(b"changed player")
            with self.assertRaisesRegex(ValueError,"differs"):
                verify(player)

    def test_unity_python_contract(self):
        source = (Path(__file__).resolve().parents[2] /
                  "Assets/Scripts/HorseTrainingObservation.cs").read_text(encoding="utf-8")
        constants = dict((key, int(value)) for key, value in
                         re.findall(r"public const int (\w+)=(\d+);", source))
        self.assertEqual(constants["ProprioFrame"], FRAME)
        self.assertEqual(constants["History"], HISTORY)
        self.assertEqual(constants["Terrain"], 70)
        self.assertEqual(constants["Opponent"], 12)
        self.assertEqual(OBSERVATION, FRAME * HISTORY + 70 + 36)
        self.assertEqual(ACTION, 28)
        self.assertIn(f'PhysicsFingerprint="{PHYSICS_FINGERPRINT}"',source)

    def test_reward_and_course_contract_match_unity_and_hud(self):
        root=Path(__file__).resolve().parents[2]
        reward=(root/"Assets/Scripts/HorseTrainingReward.cs").read_text(encoding="utf-8")
        area=(root/"Assets/Scripts/Training/HorseTrainingArea.cs").read_text(encoding="utf-8")
        reward_text=re.search(r'const string Contract="([^"]+)"',reward).group(1)
        course_text=re.search(r'const string CourseContract="([^"]+)"',area).group(1)
        self.assertEqual(hashlib.sha256(reward_text.encode()).hexdigest(),REWARD_HASH)
        self.assertEqual(hashlib.sha256(course_text.encode()).hexdigest(),COURSE_HASH)
        goals=json.loads((root/"Assets/Resources/QHopStageGoals.json").read_text(encoding="utf-8"))
        self.assertEqual([(row["distance"],row["timeLimit"])
                          for row in goals["stage1Tasks"]],
                         [(distance,seconds) for _,distance,seconds in STAGE1_TASKS])

    def test_binary_channels_untouched(self):
        mask = normalization_mask()
        for frame in range(HISTORY):
            base = frame * FRAME
            for i in range(4):
                self.assertEqual(mask[base + 174 + i * 4], 0)
            self.assertTrue(np.all(mask[base + 205:base + 209] == 0))
            self.assertTrue(np.all(mask[base + 210:base + 212] == 0))


class CurriculumTests(unittest.TestCase):
    def test_episode_summary_uses_eligible_meters_and_dnf_reason(self):
        from qhop_train.runner import episode_summary
        sample={"progress":2.0,"progressMeters":2.4,
                "weightedProgress":4.0,"finished":False,"dnf":True,
                "reason":"timeout","falls":1,"stationaryFraction":.6,
                "filteredGround":.5}
        summary=episode_summary([sample])
        self.assertEqual(summary["progress_m"],2.4)
        self.assertEqual(summary["reward_progress"],4.0)
        self.assertEqual(summary["dnf_reasons"],{"timeout":1})
        self.assertEqual(summary["filtered_ground_m"],.5)

    def test_budgets_and_final_shaping(self):
        self.assertEqual(sum(phase.budget for phase in PHASES), 63_000_000)
        self.assertEqual([difficulty(3, v) for v in (0, .25, .5, .75)],
                         [.25, .5, .75, 1.0])
        self.assertEqual([shaping(6, v) for v in (0, .2, .4, .6, .8)],
                         [1.0, .75, .5, .25, 0.0])

    def test_stage1_short_tasks_and_final_review_gate(self):
        self.assertEqual([(distance,seconds) for _,distance,seconds in STAGE1_TASKS],
                         [(2,10),(5,15),(10,20),(20,20)])
        self.assertEqual(evaluation_interval(1),25)
        self.assertEqual(evaluation_interval(2),100)
        self.assertEqual(task(1,2)[1:],(10,20))
        self.assertEqual(stage1_result(0,{"completion":.79,"fall_free":1},0),(0,0,False))
        self.assertEqual(stage1_result(0,{"completion":.8,"fall_free":.6},0),(1,0,False))
        for count in (0,1):
            self.assertEqual(stage1_result(3,{"completion":.9,"fall_free":.8},count),
                             (3,count+1,False))
        self.assertEqual(stage1_result(3,{"completion":.9,"fall_free":.8},2),(3,3,True))
        self.assertEqual(stage1_result(3,{"completion":.89,"fall_free":.8},2),(3,0,False))
        self.assertLessEqual(6*25*16384,PHASES[0].budget)

    def test_stage6_requires_evaluation_and_final_training_span(self):
        budget=10_000_000
        self.assertFalse(stage6_can_advance(0,999_999,0,budget,.95,.95))
        self.assertFalse(stage6_can_advance(0,1_000_000,0,budget,.89,.95))
        self.assertTrue(stage6_can_advance(0,1_000_000,0,budget,.90,.95))
        self.assertFalse(stage6_can_advance(1,2_000_000,1_500_000,budget,.90,.90))
        self.assertFalse(stage6_final_ready(4,9_999_999,8_000_000,budget))
        self.assertTrue(stage6_final_ready(4,10_000_000,8_000_000,budget))


class ControlTests(unittest.TestCase):
    def test_stop_request_is_scoped_to_session_and_status_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            state={"stage": 2, "global_updates": 7, "review": {"version": 1, "status": "training"}}
            session=Session(directory,state)
            request_stop(directory,"another-session")
            session.check_stop()
            request_stop(directory,session.id)
            with self.assertRaises(StopRequested):
                session.check_stop()
            session.publish("stopped")
            report=json.loads((Path(directory)/"session-status.json").read_text())
            self.assertEqual(report["updates"],7)
            self.assertEqual(report["status"],"stopped")

    def test_manual_gate_needs_three_passes_exact_hash_and_complete_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate=Path(directory)/"candidate.pt"
            candidate.write_bytes(b"frozen model")
            training={"stage": 1, "passes": 2}
            with self.assertRaisesRegex(ValueError,"Three consecutive"):
                freeze_candidate(training,candidate,{"completion":1.0})
            training["passes"]=3
            freeze_candidate(training,candidate,{"completion":1.0})
            with self.assertRaisesRegex(ValueError,"complete preview"):
                approve(training,candidate)
            with self.assertRaisesRegex(ValueError,"At least one"):
                record_preview(training,candidate,92500,0)
            record_preview(training,candidate,92500,1)
            changed=Path(directory)/"other.pt"
            changed.write_bytes(b"another model")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                approve(training,changed)
            approve(training,candidate,"2026-01-01T00:00:00Z")
            self.assertEqual(review_state(training)["status"],"approved")

    def test_legacy_state_is_unreviewed(self):
        self.assertEqual(review_state({})["status"],"training")


try:
    import torch
    from qhop_train.checkpoint import load, save
    from qhop_train.model import Actor
    from qhop_train.ppo import PPO, PhaseCosine, Rollout
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class PPOTests(unittest.TestCase):
    def test_failed_partial_optimizer_update_keeps_last_complete_checkpoint(self):
        from qhop_train import runner
        with tempfile.TemporaryDirectory() as directory:
            ppo=PPO("cpu",total_updates=20)
            before=[parameter.detach().clone() for parameter in ppo.actor.parameters()]
            class FakeWorkers:
                def __init__(self,*args,**kwargs):pass
                def __enter__(self):return self
                def __exit__(self,*args):pass
            def broken_update(*args,**kwargs):
                next(ppo.actor.parameters()).data.add_(1)
                raise RuntimeError("synthetic mid-update failure")
            with patch.object(runner,"PPO",return_value=ppo), \
                 patch.object(runner,"UnityWorkers",FakeWorkers), \
                 patch.object(runner,"_build_digest",return_value="test"), \
                 patch.object(runner,"collect",return_value=(object(),
                     np.zeros((1,OBSERVATION),np.float32),0,[])), \
                 patch.object(ppo,"update",side_effect=broken_update):
                with self.assertRaisesRegex(RuntimeError,"mid-update"):
                    runner.train("synthetic-player",directory,worker_count=1,
                                 batch_size=128,max_updates=1)
            saved=PPO("cpu")
            payload=load(Path(directory)/"latest.pt",saved)
            self.assertEqual(payload["training"]["global_updates"],0)
            for expected,actual in zip(before,saved.actor.parameters()):
                torch.testing.assert_close(expected,actual)
            report=json.loads((Path(directory)/"session-status.json").read_text())
            self.assertEqual(report["status"],"failed")

    def test_stop_requested_during_update_commits_that_update(self):
        from qhop_train import runner
        with tempfile.TemporaryDirectory() as directory:
            ppo=PPO("cpu",total_updates=20)
            class FakeWorkers:
                def __init__(self,*args,**kwargs):pass
                def __enter__(self):return self
                def __exit__(self,*args):pass
            class FakeRollout:
                reward=np.zeros((1,1),np.float32)
                action=np.zeros((1,1,ACTION),np.float32)
                actor_mask=np.ones((1,1),np.float32)
            def update_then_stop(*args,**kwargs):
                ppo.schedule.advance()
                status=json.loads((Path(directory)/"session-status.json").read_text())
                request_stop(directory,status["session_id"])
                return {"lr":0.0}
            with patch.object(runner,"PPO",return_value=ppo), \
                 patch.object(runner,"UnityWorkers",FakeWorkers), \
                 patch.object(runner,"_build_digest",return_value="test"), \
                 patch.object(runner,"collect",return_value=(FakeRollout(),
                     np.zeros((1,OBSERVATION),np.float32),0,[])), \
                 patch.object(ppo,"update",side_effect=update_then_stop):
                state=runner.train("synthetic-player",directory,worker_count=1,
                                   batch_size=128)
            self.assertEqual(state["global_updates"],1)
            saved=PPO("cpu")
            payload=load(Path(directory)/"latest.pt",saved)
            self.assertEqual(payload["training"]["global_updates"],1)
            self.assertEqual(saved.schedule.update,1)
            status=json.loads((Path(directory)/"session-status.json").read_text())
            self.assertEqual(status["status"],"stopped")

    def test_offline_stage_approval_resets_only_stage_schedule(self):
        from qhop_train.review import approve_stage, load_run, save_preview_evidence
        from qhop_train.control import freeze_candidate
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"stage1-candidate.pt"
            latest=Path(directory)/"latest.pt"
            ppo=PPO("cpu",total_updates=200)
            ppo.schedule.update=17
            training={"stage":1,"passes":3,"global_updates":17,
                      "phase_steps":128,"global_steps":128,
                      "phase_budget_extension":0,"opponents":[],"history":[],
                      "review":{"version":1,"status":"training"}}
            metadata={"batch":128,"workers":1,"seed":9,"unity_build":"test",
                      "physics_fingerprint":PHYSICS_FINGERPRINT,
                      "reward_hash":REWARD_HASH,"course_hash":COURSE_HASH}
            save(source,ppo,training,metadata)
            freeze_candidate(training,source,{"completion":1.0})
            save(latest,ppo,training,metadata)
            with self.assertRaisesRegex(ValueError,"complete preview"):
                approve_stage(directory)
            save_preview_evidence(directory,source,92500,1,[{"finished":False}])
            advanced=approve_stage(directory)
            self.assertEqual(advanced["stage"],2)
            self.assertEqual(advanced["review"]["status"],"training")
            self.assertEqual(len(advanced["approvals"]),1)
            restored, state, _=load_run(directory)
            self.assertEqual(restored.schedule.update,0)
            self.assertEqual(state["global_updates"],17)
            self.assertEqual(state["review"]["previous_approval"]["preview_episodes"],1)
            with self.assertRaises(ValueError):
                approve_stage(directory)

    def test_reject_candidate_keeps_optimizer_and_phase_learning_rate(self):
        from qhop_train.review import load_run, reject_and_continue
        with tempfile.TemporaryDirectory() as directory:
            ppo=PPO("cpu",total_updates=200)
            ppo.schedule.update=23
            state={"stage":1,"passes":3,"global_updates":23,
                   "phase_budget_extension":0,
                   "review":{"version":1,"status":"training"}}
            metadata={"batch":128,"unity_build":"test"}
            candidate=Path(directory)/"stage1-candidate.pt"
            save(candidate,ppo,state,metadata)
            freeze_candidate(state,candidate,{"completion":1.0})
            save(Path(directory)/"latest.pt",ppo,state,metadata)
            rejected=reject_and_continue(directory,extra_steps=1000)
            self.assertEqual(rejected["passes"],0)
            self.assertEqual(rejected["phase_budget_extension"],1000)
            restored,stored,_=load_run(directory)
            self.assertEqual(restored.schedule.update,23)
            self.assertEqual(stored["review"]["status"],"training")
    def test_scheduler_and_resume(self):
        schedule = PhaseCosine(3e-4, 1000)
        self.assertEqual(schedule.rate(), 0)
        schedule.update = schedule.warmup
        self.assertAlmostEqual(schedule.rate(), 3e-4)
        schedule.update = 1000
        self.assertAlmostEqual(schedule.rate(), 3e-5)
        copy = PhaseCosine(1e-4, 10)
        copy.load_state_dict(schedule.state_dict())
        self.assertAlmostEqual(copy.rate(), schedule.rate())

    def test_stage1_task_checkpoint_keeps_learning_rate_and_optimizer(self):
        ppo=PPO("cpu",total_updates=PHASES[0].updates())
        ppo.schedule.update=25
        state={"stage":1,"stage1_task_index":1,"passes":0,
               "phase_steps":25*16384}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"stage1.pt"
            save(path,ppo,state,{"reward_hash":REWARD_HASH,"course_hash":COURSE_HASH})
            restored=PPO("cpu")
            payload=load(path,restored)
            self.assertEqual(payload["training"]["stage1_task_index"],1)
            self.assertEqual(restored.schedule.update,25)
            self.assertAlmostEqual(restored.schedule.rate(),ppo.schedule.rate())
            self.assertEqual(restored.optimizer.state_dict()["param_groups"],
                             ppo.optimizer.state_dict()["param_groups"])

    def test_gae_separates_agents_and_terminal(self):
        rollout = Rollout(
            observation=np.zeros((4, 1, OBSERVATION), np.float32),
            action=np.zeros((4, 1, ACTION), np.float32),
            logp=np.zeros((4, 1), np.float32),
            value=np.zeros((4, 1), np.float32),
            next_value=np.zeros((4, 1), np.float32),
            reward=np.array([[1], [10], [2], [20]], np.float32),
            done=np.array([[False], [False], [True], [True]]),
            actor_mask=np.ones((4, 1), np.float32),
            stream_id=np.array([0, 1, 0, 1]))
        advantage, _ = rollout.advantages(gamma=1, gae_lambda=1)
        np.testing.assert_allclose(advantage[:, 0], [3, 30, 2, 20])

    def test_rollout_cut_bootstraps_without_crossing_terminal(self):
        rollout=Rollout(
            observation=np.zeros((2,1,OBSERVATION),np.float32),
            action=np.zeros((2,1,ACTION),np.float32),
            logp=np.zeros((2,1),np.float32),
            value=np.zeros((2,1),np.float32),
            next_value=np.array([[0],[5]],np.float32),
            reward=np.array([[1],[2]],np.float32),
            done=np.array([[True],[False]]),
            actor_mask=np.ones((2,1),np.float32))
        advantage,_=rollout.advantages(gamma=1,gae_lambda=1)
        np.testing.assert_allclose(advantage[:,0],[1,7])

    def test_actor_shapes_and_invalid_opponent_mask(self):
        torch.manual_seed(4)
        actor = Actor()
        observation = torch.zeros((2, OBSERVATION))
        action, logp = actor.sample(observation)
        self.assertEqual(tuple(action.shape), (2, ACTION))
        self.assertEqual(tuple(logp.shape), (2,))
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue((action.abs() <= 1).all())

    def test_tanh_probability_density(self):
        actor=Actor()
        mean=torch.zeros((2,ACTION))
        std=torch.ones_like(mean)*.4
        distribution=torch.distributions.Normal(mean,std)
        raw=torch.full_like(mean,.3)
        action=raw.tanh()
        expected=(distribution.log_prob(raw)-torch.log1p(-action.square())).sum(-1)
        torch.testing.assert_close(actor.log_probability(distribution,raw),expected)

    def test_penalty_samples_do_not_update_actor(self):
        ppo=PPO("cpu",total_updates=20)
        ppo.schedule.update=ppo.schedule.warmup
        rollout=Rollout(
            observation=np.zeros((16,1,OBSERVATION),np.float32),
            action=np.zeros((16,1,ACTION),np.float32),
            logp=np.zeros((16,1),np.float32),
            value=np.zeros((16,1),np.float32),
            next_value=np.zeros((16,1),np.float32),
            reward=np.ones((16,1),np.float32),
            done=np.zeros((16,1),bool),
            actor_mask=np.zeros((16,1),np.float32))
        before=[parameter.clone() for parameter in ppo.actor.parameters()]
        ppo.update(rollout,minibatch_size=16,epochs=1)
        for old,new in zip(before,ppo.actor.parameters()):
            torch.testing.assert_close(old,new)

    def test_full_checkpoint_restores_optimizer_and_learning_rate(self):
        torch.manual_seed(9)
        ppo = PPO("cpu", total_updates=20)
        rollout = Rollout(
            observation=np.zeros((16, 1, OBSERVATION), np.float32),
            action=np.zeros((16, 1, ACTION), np.float32),
            logp=np.zeros((16, 1), np.float32),
            value=np.zeros((16, 1), np.float32),
            next_value=np.zeros((16, 1), np.float32),
            reward=np.ones((16, 1), np.float32),
            done=np.zeros((16, 1), dtype=bool),
            actor_mask=np.ones((16, 1), np.float32))
        ppo.update(rollout, minibatch_size=16, epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.pt"
            save(path, ppo, {"stage": 1}, {"unity_build": "test"})
            restored = PPO("cpu", total_updates=20)
            load(path, restored)
            self.assertEqual(restored.schedule.update, ppo.schedule.update)
            self.assertEqual(restored.optimizer.state_dict()["state"].keys(),
                             ppo.optimizer.state_dict()["state"].keys())
            random_state=torch.get_rng_state()
            ppo.update(rollout, minibatch_size=16, epochs=1)
            torch.set_rng_state(random_state)
            restored.update(rollout, minibatch_size=16, epochs=1)
            for first, second in zip(ppo.actor.parameters(), restored.actor.parameters()):
                torch.testing.assert_close(first, second)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_checkpoint_restores_cpu_and_gpu_rng(self):
        ppo=PPO("cuda",total_updates=20)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cuda-check.pt"
            save(path,ppo,{"stage":1},{"unity_build":"test"})
            expected_cpu=torch.get_rng_state().clone()
            expected_cuda=torch.cuda.get_rng_state().clone()
            torch.rand(1)
            torch.rand(1,device="cuda")
            load(path,ppo)
            torch.testing.assert_close(torch.get_rng_state(),expected_cpu)
            torch.testing.assert_close(torch.cuda.get_rng_state(),expected_cuda)

    @unittest.skipIf(find_spec("onnxruntime") is None, "ONNX Runtime not installed")
    def test_export_matches_pytorch_actions(self):
        import onnxruntime as ort
        from qhop_train.export import DeploymentActor, export
        torch.manual_seed(28)
        ppo=PPO("cpu")
        raw=torch.randn(4,OBSERVATION)
        expected=DeploymentActor(ppo.actor.eval(),ppo.normalizer.mean,
                                 ppo.normalizer.var)(raw).detach().numpy()
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"checkpoint.pt"
            model=Path(directory)/"QHopPolicy.onnx"
            save(source,ppo,{"stage":1},{"unity_build":"test",
                                           "physics_fingerprint":PHYSICS_FINGERPRINT})
            export(source,model)
            session=ort.InferenceSession(str(model),providers=["CPUExecutionProvider"])
            actual=session.run(["action"],{"observation":raw.numpy()})[0]
            self.assertLess(float(np.max(np.abs(actual-expected))),1e-4)

    def test_export_rejects_old_physics_contract(self):
        from qhop_train.export import export
        ppo=PPO("cpu")
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"old.pt"
            save(source,ppo,{"stage":1},{"physics_fingerprint":"stale"})
            with self.assertRaisesRegex(ValueError,"physics contract"):
                export(source,Path(directory)/"QHopPolicy.onnx")


if __name__ == "__main__":
    unittest.main()
