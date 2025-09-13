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

    <!-- Ground and walls -->
    <geom name="ground" type="plane" pos="0 0 0" size="50 50 1" rgba="1 1 1 1"/>
    <geom name="wall_front" type="box" pos="50 0 5" size="0.5 50 5" rgba="0.7 0.7 0.7 1"/>
    <geom name="wall_back"  type="box" pos="-50 0 5" size="0.5 50 5" rgba="0.7 0.7 0.7 1"/>
    <geom name="wall_left"  type="box" pos="0 -50 5" size="50 0.5 5" rgba="0.7 0.7 0.7 1"/>
    <geom name="wall_right" type="box" pos="0 50 5" size="50 0.5 5" rgba="0.7 0.7 0.7 1"/>
    
    <body name="chassis" pos="0 0 0.5">
      <geom name="chassis_geom" type="box" size="0.3 0.15 0.05"
            rgba="0.3 0.3 0.3 1" mass="100"/>
      <joint name="chassis_free" type="free"/>

      <!-- Lidar sites at 4 directions -->
      <site name="ray_site_front" pos="0.3 0 0.1" quat="0.7071 0 -0.7071 0" type="sphere" size="0.05"/>
      <site name="ray_site_right1" pos="0.125 0.17 0.1" quat="0.7071 0.7071 0 0" type="sphere" size="0.05"/>
      <site name="ray_site_right2" pos="-0.125 0.17 0.1" quat="0.7071 0.7071 0 0" type="sphere" size="0.05"/>
      <site name="ray_site_back" pos="-0.3 0 0.1" quat="0 -0.7071 0 -0.7071" type="sphere" size="0.05"/>
      <site name="ray_site_left1" pos="0.125 -0.17 0.1" quat="0.7071 -0.7071 0 0" type="sphere" size="0.05"/>
      <site name="ray_site_left2" pos="-0.125 -0.17 0.1" quat="0.7071 -0.7071 0 0" type="sphere" size="0.05"/>

      <!-- Rear left wheel -->
      <body name="rear_left" pos="-0.3 -0.25 0" euler="90 0 0">
        <joint name="rear_left_axle" type="hinge" axis="0 0 1" limited="false"/>
        <geom name="rear_left_wheel" type="cylinder" size="0.08 0.04"
              rgba="0.7 0.7 0.7 1" mass="20"/>
      </body>

      <!-- Rear right wheel -->
      <body name="rear_right" pos="-0.3 0.25 0" euler="90 0 0">
        <joint name="rear_right_axle" type="hinge" axis="0 0 1" limited="false"/>
        <geom name="rear_right_wheel" type="cylinder" size="0.08 0.04"
              rgba="0.7 0.7 0.7 1" mass="20"/>
      </body>

      <!-- Front left steering + wheel -->
      <body name="front_left_mount" pos="0.3 -0.25 0" euler="0 90 0">
        <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
        <joint name="front_left_steer" type="hinge" axis="1 0 0"
               limited="true" range="-20 20"/>
        <body name="front_left" pos="0 0 0" euler="90 0 0">
          <joint name="front_left_axle" type="hinge" axis="0 0 1" limited="false"/>
          <geom name="front_left_wheel" type="cylinder" size="0.08 0.04"
                rgba="0.7 0.7 0.7 1" mass="20"
                friction="1 10 1" condim="1"/>
        </body>
      </body>

      <!-- Front right steering + wheel -->
      <body name="front_right_mount" pos="0.3 0.25 0" euler="0 90 0">
        <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
        <joint name="front_right_steer" type="hinge" axis="1 0 0"
               limited="true" range="-20 20"/>
        <body name="front_right" pos="0 0 0" euler="90 0 0">
          <joint name="front_right_axle" type="hinge" axis="0 0 1" limited="false"/>
          <geom name="front_right_wheel" type="cylinder" size="0.08 0.04"
                rgba="0.7 0.7 0.7 1" mass="20"
                friction="1 10 1" condim="1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <tendon>
      <!-- Tendon that couples both rear axles -->
      <fixed name="rear_axles">
        <joint joint="rear_left_axle" coef="1"/>
        <joint joint="rear_right_axle" coef="1"/>
      </fixed>

      <!-- Tendon that couples both front steering joints -->
      <fixed name="front_steer">
        <joint joint="front_left_steer" coef="1"/>
        <joint joint="front_right_steer" coef="1"/>
      </fixed>
   </tendon>

  <actuator>
    <!-- One actuator controls both rear wheels -->
    <motor name="rear_drive" tendon="rear_axles" gear="-6" ctrlrange="0 1"/>

    <!-- One actuator controls both front steering joints -->
    <position name="front_steer_ctrl" tendon="front_steer" kp="1000" ctrlrange="-1 1"/>
  </actuator>



  <sensor>
    <rangefinder name="lidar_front" site="ray_site_front" cutoff="5"/>
    <rangefinder name="lidar_right1" site="ray_site_right1" cutoff="5"/>
    <rangefinder name="lidar_right2" site="ray_site_right2" cutoff="5"/>
    <rangefinder name="lidar_back" site="ray_site_back" cutoff="5"/>
    <rangefinder name="lidar_left1" site="ray_site_left1" cutoff="5"/>
    <rangefinder name="lidar_left2" site="ray_site_left2" cutoff="5"/>
  </sensor>

</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    steer = -0.3
    while viewer.is_running():
        # data.ctrl[0] = 0.1  
        # data.ctrl[1] = steer  
        print("1, 2, 3, 4, 5, 6:", data.sensordata)
        mujoco.mj_step(model, data)
        viewer.sync()
        t += model.opt.timestep
