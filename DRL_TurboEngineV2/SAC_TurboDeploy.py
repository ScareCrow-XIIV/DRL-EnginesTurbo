import torch
import SAC_TurboCore
import time
import os
from TurboRaceEnvironment import TurboRaceEnv
import numpy as np


def add_action_noise(action, noise_scale, action_space):
    action_tensor = action if isinstance(action, torch.Tensor) else torch.as_tensor(action, dtype=torch.float32)
    noise = noise_scale * torch.randn_like(action_tensor)
    noisy_action = action_tensor + noise

    lower_bound = torch.as_tensor(action_space.low, dtype=torch.float32)
    upper_bound = torch.as_tensor(action_space.high, dtype=torch.float32)

    clipped_action = torch.clamp(noisy_action, lower_bound, upper_bound)
    return clipped_action


def load_policy(ac, model_path):
    # In PyTorch >=2.6, explicitly disable weights_only so replay buffer dict loads
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

    if isinstance(checkpoint, dict):
        if "pi" in checkpoint:
            ac.pi.load_state_dict(checkpoint["pi"])
            print("Loaded policy from full SAC checkpoint.")
        else:
            # If checkpoint is just the actor state_dict
            ac.pi.load_state_dict(checkpoint)
            print("Loaded plain policy weights.")

        # Ignore replay buffer and other training-only keys
        if "replay_buffer" in checkpoint:
            print("Note: Checkpoint contains a replay buffer (ignored during deployment).")
    else:
        ac.pi.load_state_dict(checkpoint)
        print("Loaded plain policy weights.")

    ac.pi.eval()
    return ac


def test_sac_policy(env_class, reset_noise=0.1, action_noise_scale=0.05,
                    model_path='droneWeights.pth', num_test_episodes=10):
    print(f"Attempting to load SAC policy from: {model_path}")

    env = None
    all_episode_returns = []

    try:
        env = env_class(render_mode="human", reset_noise_scale=reset_noise)
        ac = SAC_TurboCore.MLPActorCritic(env.observation_space, env.action_space)

        # Load trained actor from checkpoint
        ac = load_policy(ac, model_path)

        print("SAC Policy loaded successfully. Starting testing with action noise...")
        print(f"Action noise scale: {action_noise_scale}")

        for i in range(num_test_episodes):
            o, info = env.reset()
            ep_ret = 0
            ep_len = 0
            done = False
            truncated = False
            max_steps = 35000

            print(f"\n--- Running Episode {i+1} ---")
            while not (done or truncated or ep_len >= max_steps):
                a_raw = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic=True)
                a_noisy_tensor = add_action_noise(a_raw, action_noise_scale, env.action_space)
                a = a_noisy_tensor.cpu().numpy()

                o, r, done, truncated, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                env.render()
                # time.sleep(0.01)  # optional slow-down for visualization

                print(f"Step {ep_len}: "
                      f"Lidars: {o[:6]}, "
                      f"VelX: {o[6]:.3f}, VelY: {o[7]:.3f}, "
                      f"AccX: {o[8]:.3f}, AccY: {o[9]:.3f}, "
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
        '/Users/venky/Documents/Projects/DRL_TurboEngine/DRL_TurboEngineV2/'
        'SAC_TrainedWeights/SAC_TrainedWeights_TR1_epoch8.pth'
    )

    if not os.path.exists(sac_model_path):
        print(f"Error: Model file not found at {sac_model_path}")
        print("Please ensure you have run the training script and the path is correct.")
    else:
        test_sac_policy(env_class=TurboRaceEnv,
                        reset_noise=0,
                        action_noise_scale=0,
                        model_path=sac_model_path,
                        num_test_episodes=50)
