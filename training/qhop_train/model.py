"""Small feed-forward actor and critic for a 20 Hz articulated controller."""
import math

import torch
from torch import nn
from torch.distributions import Normal

from .schema import ACTION, FRAME, HISTORY, OPPONENT, OPPONENTS, SLICES, TERRAIN


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        linear = nn.Linear(a, b)
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ELU()))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_net = mlp((FRAME * HISTORY, 256, 256))
        self.terrain_net = mlp((TERRAIN, 64, 64))
        self.opponent_net = mlp((OPPONENT - 1, 32, 32))
        self.fusion = mlp((256 + 64 + 64, 256, 128))

    def forward(self, obs):
        own = self.self_net(obs[..., SLICES.proprio])
        ground = self.terrain_net(obs[..., SLICES.terrain])
        foes = obs[..., SLICES.opponents].reshape(-1, OPPONENTS, OPPONENT)
        valid = foes[..., -1:].clamp(0, 1)
        features = self.opponent_net(foes[..., :-1]) * valid
        mean = features.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        max_features = features.masked_fill(valid == 0, -1e9).max(dim=1).values
        max_features = torch.where(valid.sum(dim=1) > 0, max_features, 0)
        return self.fusion(torch.cat((own, ground, mean, max_features), dim=-1))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mean = nn.Linear(128, ACTION)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        with torch.no_grad():
            self.mean.bias[16:] = torch.atanh(torch.tensor(0.7))
        self.log_std = nn.Parameter(torch.cat((torch.full((16,), -1.4), torch.full((12,), -1.8))))

    def distribution(self, obs):
        mean = self.mean(self.encoder(obs))
        std = self.log_std.clamp(-4, 0.5).exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs):
        return self.mean(self.encoder(obs)).tanh()

    def sample(self, obs):
        dist = self.distribution(obs)
        raw = dist.rsample()
        action = raw.tanh()
        return action, self.log_probability(dist, raw)

    @staticmethod
    def log_probability(dist, raw):
        # Numerically stable log(1-tanh(x)^2).
        correction = 2 * (math.log(2) - raw - nn.functional.softplus(-2 * raw))
        return (dist.log_prob(raw) - correction).sum(dim=-1)

    def evaluate(self, obs, action):
        raw = torch.atanh(action.clamp(-0.999999, 0.999999))
        dist = self.distribution(obs)
        logp = self.log_probability(dist, raw)
        sampled = dist.rsample()
        entropy = -self.log_probability(dist, sampled)
        return logp, entropy


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.value = nn.Linear(128, 1)
        nn.init.orthogonal_(self.value.weight, gain=1)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs):
        return self.value(self.encoder(obs)).squeeze(-1)
