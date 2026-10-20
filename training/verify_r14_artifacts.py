"""Verify saved engineering fixtures without starting training or Unity."""
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from qhop_train.ppo import PPO
from qhop_train.checkpoint import load
from qhop_train.export import DeploymentActor
from qhop_train.schema import OBSERVATION, SLICES


def main():
    root = Path(__file__).resolve().parents[1]
    logs = root / 'Builds/BuildLogs'
    torch.set_num_threads(2)
    trained = torch.load(root / 'Runs/r14-validation-20261013/latest.pt', map_location='cpu', weights_only=False)
    torch.manual_seed(trained['metadata']['seed'])
    initial = PPO('cpu')
    differences = {}
    for name in ('actor', 'critic', 'vision'):
        differences[name] = max(float((value - trained[name][key]).abs().max())
                                for key, value in getattr(initial, name).state_dict().items())
    assert all(value > 0 for value in differences.values()), differences
    optimizer_steps = sorted({int(value['step']) for value in trained['optimizer']['state'].values()})
    assert trained['training']['global_updates'] == 2 and optimizer_steps
    fixture = root / 'Builds/R13ValidationProject/Assets/Editor/TestAssets'
    ppo = PPO('cpu')
    load(logs / 'r14-export-fixture.pt', ppo, weights_only=True)
    actor = DeploymentActor(ppo.actor.eval(), ppo.normalizer.mean, ppo.normalizer.var)
    sessions = []
    for name in ('QHopSmoke.onnx', 'QHopSmokeVision.onnx'):
        onnx.checker.check_model(onnx.load(fixture / name))
        sessions.append(ort.InferenceSession(str(fixture / name), providers=['CPUExecutionProvider']))
    records = json.loads((root / 'Builds/R13ValidationProject/Builds/BuildLogs/qhop-inference-probe.json').read_text())['steps']
    h = torch.zeros(2, 128)
    oh = np.zeros((2, 128), np.float32)
    hidden_error = action_error = onnx_error = 0.0
    with torch.no_grad():
        for step, record in enumerate(records):
            x = torch.tensor(record['observation']).reshape(2, OBSERVATION)
            if step == 12:
                h.zero_(); oh.fill(0)
            if step % 4 == 0:
                h = ppo.vision(x[:, SLICES.lidar], h)
                oh = sessions[1].run(None, {'scan': x[:, SLICES.lidar].numpy(), 'hidden': oh})[0]
            a = actor(x, h).numpy()
            oa = sessions[0].run(None, {'observation': x.numpy(), 'hidden': oh})[0]
            hidden_error = max(hidden_error, float(np.max(np.abs(h.numpy() - np.array(record['hidden']).reshape(2, 128)))))
            action_error = max(action_error, float(np.max(np.abs(a - np.array(record['action']).reshape(2, 28)))))
            onnx_error = max(onnx_error, float(np.max(np.abs(a - oa))), float(np.max(np.abs(h.numpy() - oh))))
    assert max(hidden_error, action_error, onnx_error) <= 1e-4
    result = dict(steps=len(records), horses=2, maximum_hidden_difference=hidden_error,
                  maximum_action_difference=action_error, maximum_onnx_difference=onnx_error,
                  trained_parameter_differences=differences, optimizer_steps=optimizer_steps,
                  updates=2, passed=True)
    (logs / 'r14-artifact-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
