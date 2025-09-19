import torch
import TurboCore
import mujoco
import time
import os
from TurboRaceEnvironment import TurboRaceEnv
import numpy as np


class LowPassFilter:
    def __init__(self, action_space, alpha=0.2):
        self.alpha = alpha
        self.prev_action = np.zeros_like(action_space.low)

    def reset(self):
        self.prev_action = None

    def filter(self, action):
        if self.prev_action is None:
            self.prev_action = action
            return action
        filtered = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = filtered
        return filtered


def add_action_noise(action, noise_scale, action_space):
    action_tensor = action if isinstance(action, torch.Tensor) else torch.as_tensor(action, dtype=torch.float32)
    noise = noise_scale * torch.randn_like(action_tensor)
    noisy_action = action_tensor + noise
    lower_bound = torch.as_tensor(action_space.low, dtype=torch.float32)
    upper_bound = torch.as_tensor(action_space.high, dtype=torch.float32)
    clipped_action = torch.clamp(noisy_action, lower_bound, upper_bound)
    return clipped_action


def load_policy(ac, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    if isinstance(checkpoint, dict):
        if "pi" in checkpoint:
            ac.pi.load_state_dict(checkpoint["pi"])
            print("Loaded policy from full SAC checkpoint.")
        else:
            ac.pi.load_state_dict(checkpoint)
            print("Loaded plain policy weights.")
        if "replay_buffer" in checkpoint:
            print("Note: Checkpoint contains a replay buffer (ignored during deployment).")
    else:
        ac.pi.load_state_dict(checkpoint)
        print("Loaded plain policy weights.")
    ac.pi.eval()
    return ac


def test_sac_policy(env_class, action_noise_scale=0.05,
                    model_path='droneWeights.pth', num_test_episodes=10,
                    smoothing_alpha=0.2):
    print(f"Attempting to load SAC policy from: {model_path}")
    env = None
    all_episode_returns = []

    try:
        env = env_class(render_mode="human")
        ac = TurboCore.MLPActorCritic(env.observation_space, env.action_space)
        ac = load_policy(ac, model_path)
        action_filter = LowPassFilter(env.action_space, alpha=smoothing_alpha)
        print("SAC Policy loaded successfully. Starting testing with action noise and low-pass filtering...")
        print(f"Action noise scale: {action_noise_scale}, Low-pass alpha: {smoothing_alpha}")

        for i in range(num_test_episodes):
            o, info = env.reset()
            action_filter.reset()
            ep_ret = 0
            ep_len = 0
            done = False
            truncated = False
            max_steps = 25000
            print(f"\n--- Running Episode {i+1} ---")

            while not (done or truncated or ep_len >= max_steps):
                a_raw = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic=True)
                a_noisy_tensor = add_action_noise(a_raw, action_noise_scale, env.action_space)
                a_noisy = a_noisy_tensor.cpu().numpy()
                a_filtered = action_filter.filter(a_noisy)
                o, r, done, truncated, info = env.step(a_filtered)
                ep_ret += r
                ep_len += 1
                env.render()
                #time.sleep(0.01)

                if hasattr(env, "viewer") and env.viewer is not None:
                    env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    env.viewer.cam.trackbodyid = env.model.body("chassis").id
                    env.viewer.cam.elevation = -90
                    env.viewer.cam.azimuth = -90

                print(f"Step {ep_len}: "
                            #f"Lidars: {o[:6]}, "
                            f"VelX: {o[6]:.3f}, VelY: {o[7]:.3f}, "
                            f"AccX: {o[8]:.3f}, AccY: {o[9]:.3f}, "
                            f"ChassisX: {info['chassis_xy'][0]:.3f}, "
                            f"ChassisY: {info['chassis_xy'][1]:.3f}, "
                            f"Steps: {info['steps_to_goal']}, "
                            f"G_Dis: {info['goal_distance']}, "
                            f"Truncated: {truncated}")
                
            print(f"Episode {i+1} finished: Return = {ep_ret:.3f}, Length = {ep_len} steps.")
            all_episode_returns.append(ep_ret)

        if all_episode_returns:
            mean_return = np.mean(all_episode_returns)
            std_return = np.std(all_episode_returns)
            print("\n--- Test Results ---")
            print(f"Mean Episode Return over {num_test_episodes} episodes: {mean_return:.3f}")
            print(f"Standard Deviation of Episode Returns: {std_return:.3f}")
        else:
            print("\nNo episodes completed to calculate statistics.")
    except KeyboardInterrupt:
        print("\nTesting interrupted by user (KeyboardInterrupt).")
    except FileNotFoundError:
        print(f"\nError: Model file not found at {model_path}. Please check the path.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
            print("Environment closed.")
        print("Testing process finished.")


if __name__ == '__main__':
    sac_model_path = (
        '/Users/venky/Documents/Projects/DRL_TurboEngine/DRLTurboEngines/DRL_TurboEngine_V2/'
        'SavedWeights/TrainedWeights_TR2_epoch6.pth'
    )
    if not os.path.exists(sac_model_path):
        print(f"Error: Model file not found at {sac_model_path}")
        print("Please ensure you have run the training script and the path is correct.")
    else:
        test_sac_policy(env_class=TurboRaceEnv,
                        action_noise_scale=0,
                        model_path=sac_model_path,
                        num_test_episodes=5,
                        smoothing_alpha=0.2)