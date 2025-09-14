import numpy as np
import gymnasium
import mujoco
from mujoco import viewer
from gymnasium import spaces
from typing import Optional
import time
from gymnasium.wrappers import TimeLimit

TURBO_XML = """
<mujoco model="TurboV1">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.01"/>

  <default>
    <joint armature="0.01" damping="1" limited="true"/>
  </default>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="50 50 1" rgba="1 1 1 1"/>
    <geom name="wall_front" type="box" pos="50 0 5" size="0.5 50 5" rgba="0.7 0.7 0.7 1"/>
    <geom name="wall_back"  type="box" pos="-50 0 5" size="0.5 50 5" rgba="0.7 0.7 0.7 1"/>
    <geom name="wall_left"  type="box" pos="0 -50 5" size="50 0.5 5" rgba="0.7 0.7 0.7 1"/>
    <geom name="wall_right" type="box" pos="0 50 5" size="50 0.5 5" rgba="0.7 0.7 0.7 1"/>

    <geom name="track11"  type="box" pos="0 -1.5 0" size="9.5 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track12" type="box" pos="0 1.5 0" size="7 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track21" type="box" pos="9.5 7.5 0" size="0.1 9 0.5" rgba="0.2 0.2 0.2 2"/>
    <geom name="track22"  type="box" pos="7 10.5 0" size="0.1 9 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track31" type="box" pos="16 22.5 0" size="0.1 9 0.5" euler="0 0 -45" rgba="0.2 0.2 0.2 2"/>
    <geom name="track32"  type="box" pos="13 23.5 0" size="0.1 7.5 0.5" euler="0 0 -52" rgba="0.2 0.2 0.2 1"/>
    <geom name="track41" type="box" pos="14 37 0" size="0.1 12 0.5" euler="0 0 45" rgba="0.2 0.2 0.2 2"/>
    <geom name="track42"  type="box" pos="12 35 0" size="0.1 9 0.5" euler="0 0 45" rgba="0.2 0.2 0.2 1"/>
    <geom name="track51"  type="box" pos="-3.5 41.5 0" size="9.5 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track52" type="box" pos="-4.5 45 0" size="11 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track61" type="box" pos="-15.5 35 0" size="0.1 9 0.5" rgba="0.2 0.2 0.2 2"/>
    <geom name="track62"  type="box" pos="-13 32 0" size="0.1 9 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track7" type="box" pos="-16.5 23 0" size="4 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track8"  type="box" pos="-20.5 32 0" size="0.1 9 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track91"  type="box" pos="-23.5 45 0" size="19.5 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track92"  type="box" pos="-35.5 41.5 0" size="15.5 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track10" type="box" pos="18 41 0" size="0.1 12 0.5" euler="0 0 45" rgba="0.2 0.2 0.2 2"/>
    <geom name="track111" type="box" pos="19 13 0" size="0.1 22 0.5" euler="0 0 -25" rgba="0.2 0.2 0.2 2"/>
    <geom name="track112" type="box" pos="16 13 0" size="0.1 17 0.5" euler="0 0 -25" rgba="0.2 0.2 0.2 2"/>
    <geom name="track121"  type="box" pos="-7 -6 0" size="18 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track131"  type="box" pos="-15 -1.5 0" size="10 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track132" type="box" pos="-15 1.5 0" size="14 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
    <geom name="track141" type="box" pos="-28 -2 0" size="0.1 5 0.5" euler="0 0 26" rgba="0.2 0.2 0.2 2"/>

    <body name="chassis" pos="0 0 0.5">
      <geom name="chassis_geom" type="box" size="0.3 0.15 0.05" rgba="0.3 0.3 0.3 1" mass="100"/>
      <joint name="chassis_free" type="free"/>

      <site name="ray_site_front" pos="0.3 0 0.1" quat="0.7071 0 -0.7071 0" type="sphere" size="0.05"/>
      <site name="ray_site_right1" pos="0.125 0.17 0.1" quat="0.7071 0.7071 0 0" type="sphere" size="0.05"/>
      <site name="ray_site_right2" pos="-0.125 0.17 0.1" quat="0.7071 0.7071 0 0" type="sphere" size="0.05"/>
      <site name="ray_site_back" pos="-0.3 0 0.1" quat="0 -0.7071 0 -0.7071" type="sphere" size="0.05"/>
      <site name="ray_site_left1" pos="0.125 -0.17 0.1" quat="0.7071 -0.7071 0 0" type="sphere" size="0.05"/>
      <site name="ray_site_left2" pos="-0.125 -0.17 0.1" quat="0.7071 -0.7071 0 0" type="sphere" size="0.05"/>
      <site name="imu_site" pos="0 0 0.3" size="0.05" rgba="0 1 0 1"/>

      <body name="rear_left" pos="-0.3 -0.25 0" euler="90 0 0">
        <joint name="rear_left_axle" type="hinge" axis="0 0 1" limited="false"/>
        <geom name="rear_left_wheel" type="cylinder" size="0.08 0.04" rgba="0.7 0.7 0.7 1" mass="20"/>
      </body>

      <body name="rear_right" pos="-0.3 0.25 0" euler="90 0 0">
        <joint name="rear_right_axle" type="hinge" axis="0 0 1" limited="false"/>
        <geom name="rear_right_wheel" type="cylinder" size="0.08 0.04" rgba="0.7 0.7 0.7 1" mass="20"/>
      </body>

      <body name="front_left_mount" pos="0.3 -0.25 0" euler="0 90 0">
        <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
        <joint name="front_left_steer" type="hinge" axis="1 0 0" limited="true" range="-20 20"/>
        <body name="front_left" pos="0 0 0" euler="90 0 0">
          <joint name="front_left_axle" type="hinge" axis="0 0 1" limited="false"/>
          <geom name="front_left_wheel" type="cylinder" size="0.08 0.04" rgba="0.7 0.7 0.7 1" mass="20" friction="1 10 1" condim="1"/>
        </body>
      </body>

      <body name="front_right_mount" pos="0.3 0.25 0" euler="0 90 0">
        <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
        <joint name="front_right_steer" type="hinge" axis="1 0 0" limited="true" range="-20 20"/>
        <body name="front_right" pos="0 0 0" euler="90 0 0">
          <joint name="front_right_axle" type="hinge" axis="0 0 1" limited="false"/>
          <geom name="front_right_wheel" type="cylinder" size="0.08 0.04" rgba="0.7 0.7 0.7 1" mass="20" friction="1 10 1" condim="1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <tendon>
      <fixed name="rear_axles">
        <joint joint="rear_left_axle" coef="1"/>
        <joint joint="rear_right_axle" coef="1"/>
      </fixed>

      <fixed name="front_steer">
        <joint joint="front_left_steer" coef="1"/>
        <joint joint="front_right_steer" coef="1"/>
      </fixed>
   </tendon>

  <actuator>
    <motor name="rear_drive" tendon="rear_axles" gear="-6" ctrlrange="0 1"/>
    <position name="front_steer_ctrl" tendon="front_steer" kp="100" ctrlrange="-1 1"/>
  </actuator>

  <sensor>
    <rangefinder name="lidar_front" site="ray_site_front" cutoff="5"/>
    <rangefinder name="lidar_right1" site="ray_site_right1" cutoff="5"/>
    <rangefinder name="lidar_right2" site="ray_site_right2" cutoff="5"/>
    <rangefinder name="lidar_back" site="ray_site_back" cutoff="5"/>
    <rangefinder name="lidar_left1" site="ray_site_left1" cutoff="5"/>
    <rangefinder name="lidar_left2" site="ray_site_left2" cutoff="5"/>
    <velocimeter name="imu_velocity" site="imu_site"/>
    <accelerometer name="imu_acceleration" site="imu_site"/>
  </sensor>
</mujoco>
"""


class TurboRaceEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_string: str = TURBO_XML,
                 render_mode: Optional[str] = None,
                 frame_skip: int = 1,
                 reset_noise_scale: float = 0):

        super().__init__()

        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.reset_noise_scale = reset_noise_scale

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64)

        self.render_mode = render_mode
        self.viewer = None

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

       return {"velocity_magnitude": np.linalg.norm(vel),
        "linear_velocity": vel,
        "acceleration_magnitude": np.linalg.norm(acc),
        "linear_acceleration": acc}


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        noise_qpos = self.initial_qpos + self.np_random.uniform(
            low=-self.reset_noise_scale, high=self.reset_noise_scale, size=self.model.nq
        )
        noise_qvel = self.initial_qvel + self.np_random.uniform(
            low=-self.reset_noise_scale, high=self.reset_noise_scale, size=self.model.nv
        )

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = noise_qpos
        self.data.qvel[:] = noise_qvel
        mujoco.mj_forward(self.model, self.data)

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

        lin_vel_mag = info["velocity_magnitude"]
        acc_mag = info["acceleration_magnitude"]

        reward = 0.0

        lidar1, lidar2, lidar3, lidar4, lidar5, lidar6 = observation[:6]
        vel_x, vel_y = observation[6:8]
        acc_x, acc_y = observation[8:10]
        
        forward_reward = vel_x * 10       # reward for moving forward
        lateral_penalty = -abs(vel_y) * 10 # penalize sideways sliding
        reward += forward_reward + lateral_penalty

        reward += a1*20
        reward += 4

        stall_penalty = -10.0
        if vel_x < 0.07:
            reward += stall_penalty


        terminated = False
        truncated = False

        termination_penalty = -5000.0

        min_dis = 0.9
        min_vel = 0.08

        lidars = [lidar1, lidar2, lidar3, lidar4, lidar5, lidar6]

        if min(lidars) < min_dis and vel_x < min_vel:
            terminated = True
            reward = termination_penalty
            
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
    env = TurboRaceEnv(render_mode="human")
    max_steps_0f_episode = 10000
    env = TimeLimit(env, max_episode_steps=max_steps_0f_episode)
    reward_total = 0

    unwrapped_env = env.env

    obs, info = env.reset()

    for i in range(max_steps_0f_episode):

        action = np.array([1, 0])

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        #time.sleep(0.01)
        reward_total += reward

        print(f"Step {i+1}: "
              f"Lidars: {obs[:6]}, "
              f"VelX: {obs[6]:.3f}, VelY: {obs[7]:.3f}, "
              f"AccX: {obs[8]:.3f}, AccY: {obs[9]:.3f}, "
              f"LinVel Mag: {info['velocity_magnitude']:.3f}, "
              f"Acc Mag: {info['acceleration_magnitude']:.3f}, "
              f"Terminated: {terminated}, Truncated: {truncated}")


        if terminated or truncated:
            print(f"Environment test: Episode ended after {i+1} steps.")
            print("Total Reward = ", reward_total)
            obs, info = env.reset()
            print("\n--- Resetting Environment ---\n")
    env.close()
