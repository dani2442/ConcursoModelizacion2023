import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

e_3 = torch.tensor([0., 0., 1.])


def optimize(env, agent, loss, optim, iters=10000):
    for i in range(iters):
        f = agent()
        p, v = env(f)
        l = loss(f, p, v)

        optim.zero_grad()
        l.backward(retain_graph=True)
        optim.step()

        if i%100==0:
            l1 = l.item()
            print(l1)



class Agent(nn.Module):
    def __init__(self, K):
        super().__init__()

        self.f = nn.Parameter(torch.zeros(K, 3), requires_grad=True)

    def forward(self):
        return self.f


class Environment(nn.Module):
    def __init__(self, dt: float, gamma: float, g: float, m: float, p_0: np.ndarray, v_0: np.ndarray) -> None:
        super().__init__()
        
        self.dt = dt
        self.gamma = gamma
        self.g = g
        self.m = m
        self.p_0 = torch.from_numpy(p_0)
        self.v_0 = torch.from_numpy(v_0)

    def forward(self, f: Tensor):
        K = f.shape[0]

        p = torch.zeros(K, 3)
        v = torch.zeros(K, 3)
        
        v[0] = self.v_0 + (self.dt/self.m)*f[0] - self.dt*self.g*e_3
        p[0] = self.p_0 + (self.dt/2)*(v[0] + self.v_0)

        for i in range(0, K-1):
            v[i+1] = v[i] + (self.dt/self.m)*f[i] - self.dt*self.g*e_3
            p[i+1] = p[i] + (self.dt/2)*(v[i+1] + v[i])

        return p, v


class ConstrainedLoss(nn.Module):
    def __init__(self, dt: float, gamma: float, F_max: float, alpha: float) -> None:
        super().__init__()

        self.dt = dt
        self.gamma = gamma
        self.F_max = F_max
        self.alpha = alpha

    def forward(self, f: Tensor, p: Tensor, v: Tensor):
        norm = torch.linalg.vector_norm(f, dim=1)

        # cost function
        l1 = self.gamma*self.dt*torch.sum(norm)
        
        # constraint F less than F_max
        l2 = torch.sum(F.relu(norm-self.F_max))

        # constraint alpha
        l3 = torch.linalg.vector_norm(p[:, [1, 2]], dim=1) * self.alpha - p[:, 0]
        l3 = torch.sum(F.relu(l3))

        # final destintation error
        l4 = torch.linalg.vector_norm(p[-1])

        # final velocity error
        l5 = torch.linalg.vector_norm(v[-1])

        return l1 + l2 + l3 + l4 + l5


class CostLoss(nn.Module):
    def __init__(self, dt: float, gamma: float) -> None:
        super().__init__()

        self.dt = dt
        self.gamma = gamma

    def forward(self, f: Tensor, p: Tensor, v: Tensor):
        return self.gamma*self.dt*torch.sum(torch.linalg.vector_norm(f, dim=1))


class DistanceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, f: Tensor, p: Tensor, v: Tensor):
        return torch.linalg.vector_norm(p[-1]), torch.linalg.vector_norm(v[-1])
