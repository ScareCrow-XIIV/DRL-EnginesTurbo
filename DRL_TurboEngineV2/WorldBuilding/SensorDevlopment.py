import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
  <worldbody>
    <!-- Ground -->
    <geom name="ground" type="plane" pos="0 0 0" size="3 3 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- Robot -->
    <body name="robot_body" pos="0 0 0.2">
      <geom type="cylinder" size="0.1 0.2" rgba="0.2 0.5 0.8 1"/>

      <!-- Lidar sites at 4 directions -->
      <site name="ray_site_front" pos="0.1 0 0" quat="0.7071 0 -0.7071 0" type="sphere" size="0.01"/>
      <site name="ray_site_right" pos="0 0.1 0" quat="0.5 -0.5 -0.5 -0.5" type="sphere" size="0.01"/>
      <site name="ray_site_back" pos="-0.1 0 0" quat="0 -0.7071 0 -0.7071" type="sphere" size="0.01"/>
      <site name="ray_site_left" pos="0 -0.1 0" quat="0.5 0.5 0.5 -0.5" type="sphere" size="0.01"/>
    </body>

    <!-- Walls around -->
    <geom name="wall_front" type="box" pos="1.2 0 0.25" size="0.05 1 0.25" rgba="1 0 0 1"/>
    <geom name="wall_back"  type="box" pos="-1.2 0 0.25" size="0.05 1 0.25" rgba="0 1 0 1"/>
    <geom name="wall_left"  type="box" pos="0 -1.2 0.25" size="1 0.05 0.25" rgba="0 0 1 1"/>
    <geom name="wall_right" type="box" pos="0 1.2 0.25" size="1 0.05 0.25" rgba="1 1 0 1"/>
  </worldbody>

  <sensor>
    <rangefinder name="lidar_front" site="ray_site_front" cutoff="5"/>
    <rangefinder name="lidar_right" site="ray_site_right" cutoff="5"/>
    <rangefinder name="lidar_back" site="ray_site_back" cutoff="5"/>
    <rangefinder name="lidar_left" site="ray_site_left" cutoff="5"/>
  </sensor>
</mujoco>
"""

# Load model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # Print 4 sensor readings
        print("Front, Right, Back, Left:", data.sensordata)

        viewer.sync()
        time.sleep(model.opt.timestep)
