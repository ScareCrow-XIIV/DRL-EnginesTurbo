from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import TurboCore
import os
import itertools
from TurboRaceEnvironment import TurboRaceEnv
import time


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def state_dict(self):
        return dict(obs_buf=self.obs_buf,
                    obs2_buf=self.obs2_buf,
                    act_buf=self.act_buf,
                    rew_buf=self.rew_buf,
                    done_buf=self.done_buf,
                    ptr=self.ptr,
                    size=self.size,
                    max_size=self.max_size)

    def load_state_dict(self, state_dict):
        self.obs_buf = state_dict["obs_buf"]
        self.obs2_buf = state_dict["obs2_buf"]
        self.act_buf = state_dict["act_buf"]
        self.rew_buf = state_dict["rew_buf"]
        self.done_buf = state_dict["done_buf"]
        self.ptr = state_dict["ptr"]
        self.size = state_dict["size"]
        self.max_size = state_dict["max_size"]


def sac(env_name=TurboRaceEnv, actor_critic=TurboCore.MLPActorCritic, ac_kwargs=dict(),
        seed=10, steps_per_epoch=8000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, max_ep_len=1000, 
        model_save_path='droneWeights.pth', resume_path=None, resume_full=False):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_name(render_mode="None")
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    for p in ac_targ.parameters():
        p.requires_grad = False

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=q_lr)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    start_epoch = 0
    start_step = 0

    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        ac.pi.load_state_dict(checkpoint["pi"])
        ac.q1.load_state_dict(checkpoint["q1"])
        ac.q2.load_state_dict(checkpoint["q2"])
        ac_targ.q1.load_state_dict(checkpoint["q1_targ"])
        ac_targ.q2.load_state_dict(checkpoint["q2_targ"])
        if resume_full:
            pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
            q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            if "replay_buffer" in checkpoint:
                replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        start_epoch = checkpoint.get("epoch", 0)
        start_step = start_epoch * steps_per_epoch
        print(f"\nResumed training from epoch {start_epoch}")

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        with torch.no_grad():
            a2, logp_a2 = ac.pi(o2)
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        return loss_q1 + loss_q2

    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        return (alpha * logp_pi - q_pi).mean()

    def update(data):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        for p in q_params:
            p.requires_grad = False
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        for p in q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return loss_q.item(), loss_pi.item()

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    o, info = env.reset()
    ep_ret, ep_len = 0, 0
    current_q_loss, current_pi_loss = 0.0, 0.0
    all_ep_returns_in_epoch = []

    total_env_steps = steps_per_epoch * epochs

    print(f"Starting SAC training for {epochs} epochs, with {steps_per_epoch} steps per epoch.")
    print("-"*100)

    for t in range(start_step, total_env_steps):
        current_epoch = (t + 1) // steps_per_epoch + 1
        if current_epoch == epochs and (t + 1) % steps_per_epoch == 0 and env.render_mode != 'human':
            env.close()
            env = env_name(render_mode="human")
            o, info = env.reset()

        if t >= start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_ret += r
        ep_len += 1
        done = False if ep_len == max_ep_len else done

        replay_buffer.store(o, a, r, o2, done)
        o = o2

        if done or (ep_len == max_ep_len):
            all_ep_returns_in_epoch.append(ep_ret)
            o, info = env.reset()
            ep_ret, ep_len = 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                current_q_loss, current_pi_loss = update(batch)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            avg_ep_ret = np.mean(all_ep_returns_in_epoch) if all_ep_returns_in_epoch else 0.0
            print(f"Epoch {epoch}/{epochs} | Q Loss: {current_q_loss:.3f} | Pi Loss: {current_pi_loss:.3f} | Avg Ep Return: {avg_ep_ret:.3f}")
            all_ep_returns_in_epoch = []

            save_dir = os.path.dirname(model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            epoch_model_path = model_save_path.replace(".pth", f"_epoch{epoch}.pth")
            checkpoint = {
                "pi": ac.pi.state_dict(),
                "q1": ac.q1.state_dict(),
                "q2": ac.q2.state_dict(),
                "q1_targ": ac_targ.q1.state_dict(),
                "q2_targ": ac_targ.q2.state_dict(),
                "pi_optimizer": pi_optimizer.state_dict(),
                "q_optimizer": q_optimizer.state_dict(),
                "epoch": epoch,
                "replay_buffer": replay_buffer.state_dict()
            }
            torch.save(checkpoint, epoch_model_path)
            print(f"\nPolicy model weights saved to {epoch_model_path}")

    env.close()


if __name__ == '__main__':
    sac(seed=0,
        steps_per_epoch=200000,
        epochs=500,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        alpha=0.2,
        batch_size=512,
        start_steps=70000,
        update_after=20000,
        update_every=50,
        max_ep_len=50000,
        model_save_path='/Users/venky/Documents/Projects/DRL_TurboEngine/DRLTurboEngines'
        '/DRL_TurboEngine_V2/SavedWeights/TrainedWeights_TR2_V1.pth',
        resume_path='/Users/venky/Documents/Projects/DRL_TurboEngine/DRLTurboEngines/DRL_TurboEngine_V2'
        '/SavedWeights/TrainedWeights_TR2_V1_epoch23.pth',
        resume_full=True)