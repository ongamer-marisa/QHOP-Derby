"""Small feed-forward actor and critic for a 20 Hz articulated controller."""
import math

import torch
from torch import nn
from torch.distributions import Normal

from .schema import ACTION, GAIT, FRAME, HISTORY, OPPONENT, OPPONENTS, SLICES, TERRAIN, VISION_FEATURES, VISION_HIDDEN


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
        self.gait_adapter = nn.Linear(GAIT, 128, bias=False)
        nn.init.zeros_(self.gait_adapter.weight)

    def forward(self, obs):
        own = self.self_net(obs[..., SLICES.proprio])
        ground = self.terrain_net(obs[..., SLICES.terrain])
        foes = obs[..., SLICES.opponents].reshape(-1, OPPONENTS, OPPONENT)
        valid = foes[..., -1:].clamp(0, 1)
        features = self.opponent_net(foes[..., :-1]) * valid
        mean = features.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        max_features = features.masked_fill(valid == 0, -1e9).max(dim=1).values
        max_features = torch.where(valid.sum(dim=1) > 0, max_features, 0)
        return self.fusion(torch.cat((own, ground, mean, max_features), dim=-1)) + self.gait_adapter(obs[..., SLICES.gait])


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.visual_fusion = mlp((257,128))
        self.mean = nn.Linear(128, ACTION)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        with torch.no_grad():
            self.mean.bias[16:] = torch.atanh(torch.tensor(0.7))
        self.log_std = nn.Parameter(torch.cat((torch.full((16,), -1.4), torch.full((12,), -1.8))))

    def features(self, obs, visual=None):
        if visual is None:visual=obs.new_zeros((len(obs),VISION_HIDDEN))
        age=obs[..., SLICES.age:SLICES.age+1]/.2
        return self.visual_fusion(torch.cat((self.encoder(obs),visual,age),-1))

    def distribution(self, obs, visual=None):
        mean = self.mean(self.features(obs,visual))
        std = self.log_std.clamp(-4, 0.5).exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs, visual=None):
        return self.mean(self.features(obs,visual)).tanh()

    def sample(self, obs, visual=None):
        dist = self.distribution(obs,visual)
        raw = dist.rsample()
        action = raw.tanh()
        return action, self.log_probability(dist, raw)

    @staticmethod
    def log_probability(dist, raw):
        # Numerically stable log(1-tanh(x)^2).
        correction = 2 * (math.log(2) - raw - nn.functional.softplus(-2 * raw))
        return (dist.log_prob(raw) - correction).sum(dim=-1)

    def evaluate(self, obs, action, visual=None):
        raw = torch.atanh(action.clamp(-0.999999, 0.999999))
        dist = self.distribution(obs,visual)
        logp = self.log_probability(dist, raw)
        sampled = dist.rsample()
        entropy = -self.log_probability(dist, sampled)
        return logp, entropy


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.visual_fusion = mlp((257,128))
        self.value = nn.Linear(128, 1)
        nn.init.orthogonal_(self.value.weight, gain=1)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs, visual=None):
        if visual is None:visual=obs.new_zeros((len(obs),VISION_HIDDEN))
        features=self.visual_fusion(torch.cat((self.encoder(obs),visual,obs[..., SLICES.age:SLICES.age+1]/.2),-1))
        return self.value(features).squeeze(-1)


class VisionMemory(nn.Module):
    """GRUCell equations expanded into deployment-supported primitive operations."""
    def __init__(self):
        super().__init__()
        self.encoder=mlp((VISION_FEATURES,256,128))
        self.gru=nn.GRUCell(128,VISION_HIDDEN)
        scale=torch.ones(VISION_FEATURES)
        for ray in range(57):scale[ray*3:ray*3+2]=25
        scale[171:174]=25;scale[176]=.2
        self.register_buffer('scale',scale)

    def forward(self, scan, hidden):
        encoded=self.encoder(scan/self.scale)
        gi=nn.functional.linear(encoded,self.gru.weight_ih,self.gru.bias_ih)
        gh=nn.functional.linear(hidden,self.gru.weight_hh,self.gru.bias_hh)
        ir,iz,inn=gi.chunk(3,-1);hr,hz,hn=gh.chunk(3,-1)
        reset=(ir+hr).sigmoid();update=(iz+hz).sigmoid()
        candidate=(inn+reset*hn).tanh()
        return candidate+update*(hidden-candidate)
