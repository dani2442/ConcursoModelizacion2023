import matplotlib.pyplot as plt
import torch


def plot_location(p, v):
    if isinstance(p, torch.Tensor) and isinstance(v, torch.Tensor):
        p = p.detach().numpy()
        v = v.detach().numpy()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(p[:,0], p[:,1], p[:,2], zdir='z', label='Trajectory')
    ax.scatter(p[:,0], p[:,1], p[:,2], zdir='z')

    ax.scatter(0,0,0, zdir='z', c='r', s=10, label='Objective')
    ax.legend()
    ax.set_xlim(-44, 55)
    ax.set_ylim(0, 55)
    ax.set_zlim(0, 105)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()