import time, numpy as np
from TurboRaceEnvironment import TurboRaceEnv

env = TurboRaceEnv(render_mode=None, reset_noise_scale=0)
o, info = env.reset()
N = 1000
t0 = time.time()
for i in range(N):
    a = env.action_space.sample()
    o, r, terminated, truncated, info = env.step(a)
    if terminated or truncated:
        o, info = env.reset()
t1 = time.time()
print(f"avg env.step() time = {(t1-t0)/N:.6f} s")
env.close()