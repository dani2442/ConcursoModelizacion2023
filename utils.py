import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch
import numpy as np


def plot_location(p, v, lambd = 0.5):
    if isinstance(p, torch.Tensor) and isinstance(v, torch.Tensor):
        p = p.detach().numpy()
        v = v.detach().numpy()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(p[:,0], p[:,1], p[:,2], zdir='z', label='Trajectory')
    ax.scatter(p[:,0], p[:,1], p[:,2], zdir='z')

    p2 = p + lambd*v
    p_combine = np.stack([p, p2]).transpose((1, 0, 2))
    lines = Line3DCollection(p_combine, color='r', label='Thrust')
    ax.add_collection(lines)


    ax.scatter(0,0,0, zdir='z', c='g', s=100, label='Objective')
    ax.legend()
    ax.set_xlim(-44, 55)
    ax.set_ylim(0, 55)
    ax.set_zlim(0, 105)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()