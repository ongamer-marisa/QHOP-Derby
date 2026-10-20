"""Synchronous on-policy PPO with a phase-reset cosine learning rate."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .model import Actor, Critic, VisionMemory
from .schema import OBSERVATION, normalization_mask, SLICES, VISION_HIDDEN
from .vision_state import VisionStates


class RunningMoments:
    def __init__(self):
        self.count = 1e-4
        self.mean = np.zeros(OBSERVATION, dtype=np.float64)
        self.var = np.ones(OBSERVATION, dtype=np.float64)
        self.mask = normalization_mask()

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, OBSERVATION)
        if len(values) == 0:
            return
        batch_mean, batch_var = values.mean(0), values.var(0)
        total = self.count + len(values)
        delta = batch_mean - self.mean
        self.mean += delta * len(values) / total
        self.var = (self.var * self.count + batch_var * len(values) +
                    delta ** 2 * self.count * len(values) / total) / total
        self.count = total

    def normalize(self, values):
        values = np.asarray(values, dtype=np.float32)
        standardized = (values - self.mean) / np.sqrt(self.var + 1e-8)
        return np.where(self.mask != 0, np.clip(standardized, -10, 10), values).astype(np.float32)

    def state_dict(self):
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, value):
        if any(np.asarray(value[k]).shape != (OBSERVATION,) for k in ("mean", "var")):
            raise ValueError("Observation normalization dimensions differ")
        self.count = value["count"]
        self.mean = value["mean"]
        self.var = value["var"]


class PhaseCosine:
    def __init__(self, peak, total_updates, update=0):
        self.peak = peak
        self.total_updates = max(2, total_updates)
        self.update = update

    @property
    def warmup(self):
        return min(100, max(10, math.ceil(0.02 * self.total_updates)))

    def rate(self):
        warmup = min(self.warmup, self.total_updates - 1)
        if self.update < warmup:
            return self.peak * self.update / warmup
        fraction = min(1.0, (self.update - warmup) / (self.total_updates - warmup))
        return self.peak * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction)))

    def advance(self):
        self.update += 1

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, value):
        self.__dict__.update(value)


@dataclass
class Rollout:
    observation: np.ndarray  # [time, agent, obs], already normalized at collection time
    action: np.ndarray       # [time, agent, action]
    logp: np.ndarray         # [time, agent]
    value: np.ndarray        # [time, agent]
    next_value: np.ndarray   # [time, agent]
    reward: np.ndarray       # [time, agent]
    done: np.ndarray         # true terminal, not a time-limit rollout cut
    actor_mask: np.ndarray   # zero while the action is ignored during a fall penalty
    stream_id: np.ndarray | None = None
    vision_new: np.ndarray | None = None
    vision_prefix: list | None = None

    def advantages(self, gamma=0.999, gae_lambda=0.95):
        advantage = np.zeros_like(self.reward, dtype=np.float32)
        if self.stream_id is None:
            carry = np.zeros(self.reward.shape[1], dtype=np.float32)
            for t in range(self.reward.shape[0] - 1, -1, -1):
                alive = 1.0 - self.done[t].astype(np.float32)
                delta = self.reward[t] + gamma * self.next_value[t] * alive - self.value[t]
                carry = delta + gamma * gae_lambda * alive * carry
                advantage[t] = carry
        else:
            carry = {}
            flat = advantage.reshape(-1)
            reward = self.reward.reshape(-1)
            value = self.value.reshape(-1)
            future = self.next_value.reshape(-1)
            done = self.done.reshape(-1)
            for t in range(len(flat) - 1, -1, -1):
                key = int(self.stream_id[t])
                alive = 0.0 if done[t] else 1.0
                delta = reward[t] + gamma * future[t] * alive - value[t]
                flat[t] = delta + gamma * gae_lambda * alive * carry.get(key, 0.0)
                carry[key] = float(flat[t])
        return advantage, advantage + self.value


class PPO:
    def __init__(self, device="cpu", peak_lr=3e-4, total_updates=500):
        self.device = torch.device(device)
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.vision = VisionMemory().to(self.device)
        self.vision_states = VisionStates(self.vision,self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.vision.parameters()),
            lr=0.0, betas=(0.9, 0.999), eps=1e-5)
        self.schedule = PhaseCosine(peak_lr, total_updates)
        self.normalizer = RunningMoments()

    def update(self, rollout: Rollout, entropy_coefficient=0.005,
               minibatch_size=1024, epochs=5):
        advantage, returns = rollout.advantages()
        shape = (-1,)
        obs = torch.as_tensor(rollout.observation.reshape(-1, OBSERVATION), device=self.device)
        action = torch.as_tensor(rollout.action.reshape(-1, 28), device=self.device)
        old_logp = torch.as_tensor(rollout.logp.reshape(shape), device=self.device)
        old_value = torch.as_tensor(rollout.value.reshape(shape), device=self.device)
        target = torch.as_tensor(returns.reshape(shape), device=self.device)
        actor_mask = torch.as_tensor(rollout.actor_mask.reshape(shape), device=self.device).bool()
        eligible = advantage.reshape(-1)[rollout.actor_mask.reshape(-1).astype(bool)]
        center = eligible.mean() if len(eligible) else 0.0
        scale = eligible.std() + 1e-8 if len(eligible) else 1.0
        advantage = torch.as_tensor((advantage.reshape(-1) - center) / scale, device=self.device)
        lr = self.schedule.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        # Shuffle complete sequence blocks, never individual recurrent timesteps.
        grouped={}
        stream=rollout.stream_id if rollout.stream_id is not None else np.tile(np.arange(rollout.reward.shape[1]),rollout.reward.shape[0])
        for i,key in enumerate(stream):
            grouped.setdefault(int(key),[]).append(i)
        blocks=[]
        for indices in grouped.values():
            current=[]
            for i in indices:
                current.append(i)
                if len(current)==128 or rollout.done.reshape(-1)[i]:
                    blocks.append(current);current=[]
            if current:blocks.append(current)
        statistics=[];stop=False
        new_flags=rollout.vision_new.reshape(-1) if rollout.vision_new is not None else np.ones(len(obs),dtype=bool)
        for _ in range(epochs):
            order=torch.randperm(len(blocks)).tolist()
            for start in range(0,len(order),max(1,minibatch_size//128)):
                selected=[blocks[k] for k in order[start:start+max(1,minibatch_size//128)]]
                # Padded sequence batch: only live rows enter losses and only fresh
                # scan rows enter the recurrent cell. Padding cannot advance memory.
                histories=[]
                for block in selected:
                    hidden=torch.zeros(1,VISION_HIDDEN,device=self.device)
                    prefix=rollout.vision_prefix[block[0]] if rollout.vision_prefix is not None else ()
                    with torch.no_grad():
                        for scan in prefix:hidden=self.vision(torch.as_tensor(scan,device=self.device)[None],hidden)
                    histories.append(hidden)
                hidden=torch.cat(histories,dim=0)
                ids_list=[];memory=[]
                for t in range(max(map(len,selected))):
                    fresh=[row for row,block in enumerate(selected) if t<len(block) and new_flags[block[t]]]
                    if fresh:
                        rows=torch.as_tensor(fresh,device=self.device)
                        steps=torch.as_tensor([selected[row][t] for row in fresh],device=self.device)
                        updated=self.vision(obs[steps,SLICES.lidar],hidden[rows])
                        hidden=hidden.index_copy(0,rows,updated)
                    live=[row for row,block in enumerate(selected) if t<len(block)]
                    memory.append(hidden[live]);ids_list.extend(selected[row][t] for row in live)
                ids=torch.as_tensor(ids_list,device=self.device)
                visual=torch.cat(memory,dim=0)
                logp,entropy=self.actor.evaluate(obs[ids],action[ids],visual)
                value=self.critic(obs[ids],visual)
                ratio=(logp-old_logp[ids]).exp();good=actor_mask[ids]
                if good.any():
                    surrogate=torch.minimum(ratio*advantage[ids],ratio.clamp(.8,1.2)*advantage[ids])
                    policy_loss=-surrogate[good].mean();entropy_loss=-entropy[good].mean()
                    approx_kl=(old_logp[ids][good]-logp[good]).mean().item()
                else:
                    policy_loss=value.sum()*0;entropy_loss=value.sum()*0;approx_kl=0.
                clipped_value=old_value[ids]+(value-old_value[ids]).clamp(-.2,.2)
                value_loss=torch.maximum((value-target[ids]).square(),(clipped_value-target[ids]).square()).mean()
                loss=policy_loss+.5*value_loss+entropy_coefficient*entropy_loss
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                gradient_norm=nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.critic.parameters())+list(self.vision.parameters()),.5)
                self.optimizer.step()
                clip_fraction=((ratio[good]-1).abs()>.2).float().mean().item() if good.any() else 0.
                statistics.append((policy_loss.item(),value_loss.item(),approx_kl,-entropy_loss.item(),clip_fraction,float(gradient_norm)))
                if approx_kl>.02:stop=True;break
            if stop:break
        self.vision_states.refresh()
        self.schedule.advance()
        return {"policy_loss": float(np.mean([v[0] for v in statistics])),
                "value_loss": float(np.mean([v[1] for v in statistics])),
                "kl": float(np.mean([v[2] for v in statistics])),
                "entropy": float(np.mean([v[3] for v in statistics])),
                "clip_fraction": float(np.mean([v[4] for v in statistics])),
                "gradient_norm": float(np.mean([v[5] for v in statistics])), "lr": lr}

    def reset_phase(self, peak_lr, total_updates):
        self.schedule = PhaseCosine(peak_lr, total_updates)
