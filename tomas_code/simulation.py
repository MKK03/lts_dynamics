# Libraries
import numpy as np
from model import *
import matplotlib.pyplot as plt

# Parameters
L = 10*np.pi
N = 101
nu = 0.45546875
mesh = np.linspace(0., L, N, endpoint = False)
u0 = np.random.randn(N)
t0 = 0.
dt = 1e-1

# Instanciate a one-dimensional KSE solver
kse = KSE_1D(L, N, nu, u0, t0, dt)

# Forward the system
traj = kse.forward(n_steps = 10000, keep_traj = True)

# Libraries
import numpy as np
from model import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L = 10 * np.pi
N = 101
nu = 0.45546875
mesh = np.linspace(0., L, N, endpoint=False)
u0 = np.random.randn(N)
t0 = 0.
dt = 1e-1
n_steps = 10000

# Instanciate a one-dimensional KSE solver
kse = KSE_1D(L, N, nu, u0, t0, dt)

# Forward the system
traj = kse.forward(n_steps=n_steps, keep_traj=True)  # shape assumed (n_steps+1, N) or (n_steps, N)

# Plot of the trajectory
plt.figure(figsize=(20, 5))
plt.imshow(traj.T, cmap = "RdBu", aspect = "auto", origin = "lower")
plt.colorbar()
plt.xlabel(r'$t$')
plt.ylabel(r'$\mathbf{x}$')
plt.show()

# Animation setup
n_frames = traj.shape[0]
times = t0 + dt * np.arange(n_frames)

# Optionally subsample frames so the animation isn't absurdly long
# (10000 frames at 30fps = approx 5.5 min, adjust stride to taste)
stride = 10
frame_idx = np.arange(0, n_frames, stride)

fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(mesh, traj[0], lw=1.5, color='tab:blue')
ax.set_xlim(mesh.min(), mesh.max())
ax.set_ylim(traj.min() * 1.1, traj.max() * 1.1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x,t)$')
title = ax.set_title(f't = {times[0]:.2f}')

def update(i):
    idx = frame_idx[i]
    line.set_ydata(traj[idx])
    title.set_text(f't = {times[idx]:.2f}')
    return line, title

anim = FuncAnimation(
    fig, update, frames=len(frame_idx),
    interval=30, blit=False
)

plt.tight_layout()
plt.show()

# anim.save('kse_animation.mp4', writer='ffmpeg', fps=30, dpi=150)
# anim.save('kse_animation.gif', writer='pillow', fps=30, dpi=100)
