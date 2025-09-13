import math
import mujoco
import mujoco.viewer
import time

xml = """
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
    <geom name="track131"  type="box" pos="-15 -1.5 0" size="8 0.1 0.5" rgba="0.2 0.2 0.2 1"/>
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
    steer = 0.3
    while viewer.is_running():
        # data.ctrl[0] = 0.1
        # data.ctrl[1] = steer
        vx, vy, vz = data.sensordata[model.sensor_adr[vel_id]:model.sensor_adr[vel_id]+3]
        ax, ay, az = data.sensordata[model.sensor_adr[acc_id]:model.sensor_adr[acc_id]+3]
        print("Lidars:", data.sensordata[:6])
        print("Vel:", vx, vy, "Acc:", ax, ay)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
        t += model.opt.timestep
