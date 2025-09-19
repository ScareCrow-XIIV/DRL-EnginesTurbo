import numpy as np
import gymnasium
import mujoco
from mujoco import viewer
from gymnasium import spaces
from typing import Optional
import time
from gymnasium.wrappers import TimeLimit

TURBO_XML = "/Users/venky/Documents/Projects/DRL_TurboEngine/DRLTurboEngines/DRL_TurboEngine_V2/Tracks_XML/Track_V2.xml"


class TurboRaceEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_string: str = TURBO_XML,
                 render_mode: Optional[str] = None,
                 frame_skip: int = 5,
                 goal_x: float = -10.0,
                 goal_y: float = 0.0):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_string)
        self.data = mujoco.MjData(self.model)
        
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.viewer = None
        self.steps_to_goal = 0
        
        self.goal_xy = np.array([goal_x, goal_y], dtype=np.float64)
        self.goal_radius = 1
        self.goal_reward = 1000000.0
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64)
        
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.core_body_id = self.model.body("chassis").id

    def _get_obs(self):
        lidar = self.data.sensordata[0:6].copy()
        vel = self.data.sensordata[6:9].copy()[0:2]
        acc = self.data.sensordata[9:12].copy()[0:2]
        return np.concatenate([lidar, vel, acc]).astype(np.float64)

    def _get_info(self):
        vel = self.data.sensordata[6:9].copy()
        acc = self.data.sensordata[9:12].copy()
        x, y, z = self.data.xpos[self.core_body_id]
        goal_distance = np.linalg.norm(np.array([x, y]) - self.goal_xy)
        return {
            "velocity_magnitude": np.linalg.norm(vel),
            "linear_velocity": vel,
            "acceleration_magnitude": np.linalg.norm(acc),
            "linear_acceleration": acc,
            "chassis_position": np.array([x, y, z], dtype=np.float64), 
            "chassis_xy": np.array([x, y], dtype=np.float64),
            "steps_to_goal": self.steps_to_goal,
            "goal_distance": goal_distance
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, self.data)
        self.steps_to_goal = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        a1, a2 = action[0], action[1]
        
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()

        reward = 0.0
        terminated = False
        truncated = False

        self.steps_to_goal += 1
        x, y, z = self.data.xpos[self.core_body_id]

        lidar1, lidar2, lidar3, lidar4, lidar5, lidar6 = observation[:6]
        vel_x, vel_y = observation[6:8]
        acc_x, acc_y = observation[8:10]
        
        forward_reward = -abs(vel_x) * 10
        lateral_penalty = -abs(vel_y) * 10
        reward += forward_reward + lateral_penalty
        
        if abs(vel_x) < 0.1:
            reward -= 50.0
            
        if lidar4 >= 4.9 and abs(a2) > 0.07 and vel_x > 0.2:
            reward += 10

        min_dis = 0.9
        min_vel = 0.08
        if min([lidar1, lidar2, lidar3, lidar4, lidar5, lidar6]) < min_dis and vel_x < min_vel:
            terminated = True
            reward = -5000.0

        goal_distance = np.linalg.norm(np.array([x, y]) - self.goal_xy)
        if goal_distance <= self.goal_radius:
            terminated = True
            k = 0.005  
            reward += self.goal_reward * np.exp(-k * self.steps_to_goal)
            print(f"Goal reached in {self.steps_to_goal} steps! Reward bonus: {self.goal_reward * np.exp(-k * self.steps_to_goal):.2f}")


        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.render()
            return self.viewer.read_pixels(640, 480, depth=False)

        elif self.render_mode == "human":
            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = TurboRaceEnv(render_mode="human", goal_x=-5.0, goal_y=0.0)
    max_steps_of_episode = 30000
    env = TimeLimit(env, max_episode_steps=max_steps_of_episode)
    reward_total = 0

    obs, info = env.reset()

    for i in range(max_steps_of_episode):
        action = np.array([1, 0])
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        #time.sleep(0.1)
        reward_total += reward

        print(f"Step {i+1}: "
              #f"Lidars: {obs[:6]}, "
              f"VelX: {obs[6]:.3f}, VelY: {obs[7]:.3f}, "
              f"AccX: {obs[8]:.3f}, AccY: {obs[9]:.3f}, "
              #f"LinVel Mag: {info['velocity_magnitude']:.3f}, "
              #f"Acc Mag: {info['acceleration_magnitude']:.3f}, "
              f"ChassisX: {info['chassis_xy'][0]:.3f}, "
              f"ChassisY: {info['chassis_xy'][1]:.3f}, "
              f"GoalDist: {info['goal_distance']:.3f}, "
              f"Steps: {info['steps_to_goal']}, "
              f"Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            print(f"Episode ended after {i+1} steps. Total Reward = {reward_total}")
            obs, info = env.reset()
            reward_total = 0
            print("\n--- Resetting Environment ---\n")

    env.close()
