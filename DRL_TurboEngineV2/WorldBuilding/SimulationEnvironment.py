import math
import mujoco
import mujoco.viewer
import time

xml = """
<mujoco model="TurboV1">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.02" />

  <default>
    <joint armature="0.01" damping="1" limited="true"/>
  </default>

  <asset>
  <hfield name="track_hf" file="/Users/venky/Documents/Projects/DRL_TurboEngine/DRL_TurboEngineV2/Tracks_PNG/TrackV1.png" size="50 50 5 0.1"/>
  </asset>

  <worldbody>
    <geom name="ground Plane" type="plane" pos="0 0 0" size="90 90 1" rgba="1 1 1 1" contype="1" conaffinity="1"/> 
    <geom name="wall_boundary1" type="box" pos="90 0 5" size="0.5 90 5" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
    <geom name="wall_boundary2" type="box" pos="-90 0 5" size="0.5 90 5" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
    <geom name="wall_boundary3" type="box" pos="0 -90 5" size="90 0.5 5" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
    <geom name="wall_boundary4" type="box" pos="0 90 5" size="90 0.5 5" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>

    <geom name="track_geom" type="hfield" hfield="track_hf" pos="0 -7.5 -0.5"/>

    <body name="chassis" pos="0 0 3">
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
        <geom name="rear_left_wheel" type="cylinder" size="0.08 0.04" rgba="0.7 0.7 0.7 1" mass="20" contype="1" conaffinity="1"/>
      </body>

      <body name="rear_right" pos="-0.3 0.25 0" euler="90 0 0">
        <joint name="rear_right_axle" type="hinge" axis="0 0 1" limited="false"/>
        <geom name="rear_right_wheel" type="cylinder" size="0.08 0.04" rgba="0.7 0.7 0.7 1" mass="20" contype="1" conaffinity="1"/>
      </body>

      <body name="front_left_mount" pos="0.3 -0.25 0" euler="0 90 0">
        <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
        <joint name="front_left_steer" type="hinge" axis="1 0 0" limited="true" range="-20 20"/>
        <body name="front_left" pos="0 0 0" euler="90 0 0">
          <joint name="front_left_axle" type="hinge" axis="0 0 1" limited="false"/>
          <geom name="front_left_wheel" type="cylinder" size="0.08 0.04" rgba="0.7 0.7 0.7 1" mass="20" friction="1 10 1" condim="1" contype="1" conaffinity="1"/>
        </body>
      </body>

      <body name="front_right_mount" pos="0.3 0.25 0" euler="0 90 0">
        <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
        <joint name="front_right_steer" type="hinge" axis="1 0 0" limited="true" range="-20 20"/>
        <body name="front_right" pos="0 0 0" euler="90 0 0">
          <joint name="front_right_axle" type="hinge" axis="0 0 1" limited="false"/>
          <geom name="front_right_wheel" type="cylinder" size="0.08 0.04" rgba="0.7 0.7 0.7 1" mass="20" friction="1 10 1" condim="1" contype="1" conaffinity="1"/>
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
    <motor name="rear_drive" tendon="rear_axles" gear="-50" ctrlrange="0 1"/>
    <position name="front_steer_ctrl" tendon="front_steer" kp="750" ctrlrange="-1 1"/>
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

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_velocity")
acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acceleration")

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        vx, vy, vz = data.sensordata[model.sensor_adr[vel_id]:model.sensor_adr[vel_id]+3]
        ax, ay, az = data.sensordata[model.sensor_adr[acc_id]:model.sensor_adr[acc_id]+3]
        print("Lidars:", data.sensordata[:6])
        print("Vel:", vx, vy, "Acc:", ax, ay)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
        t += model.opt.timestep