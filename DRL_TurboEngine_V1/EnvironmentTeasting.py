import math
import mujoco
import mujoco.viewer
import time

xml = "/Users/venky/Documents/Projects/DRL_TurboEngine/DRLTurboEngines/DRL_TurboEngine_V1/Tracks_XML/Track_V1.xml"

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_velocity")
acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acceleration")
with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    steer = 0
    while viewer.is_running():
        #data.ctrl[0] = 1
        #data.ctrl[1] = steer
        vx, vy, vz = data.sensordata[model.sensor_adr[vel_id]:model.sensor_adr[vel_id]+3]
        ax, ay, az = data.sensordata[model.sensor_adr[acc_id]:model.sensor_adr[acc_id]+3]
        print("Lidars:", data.sensordata[:6])
        print("Vel:", vx, vy, "Acc:", ax, ay)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
        t += model.opt.timestep
