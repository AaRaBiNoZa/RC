import time

import mujoco
from mujoco import viewer
import numpy as np

def go_to(qpos):
    data.ctrl = qpos
    for i in range(10):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        viewer_window.sync()
        if np.linalg.norm(data.qpos - qpos) < 0.0001:
            break

def start_sim():
    global model, data, viewer_window, renderer

    xml_path = "./manipulator3d.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer_window = mujoco.viewer.launch_passive(model, data)
    data.ctrl[0] = -0.2
    data.ctrl[1] = 0.2  # .00001
    data.ctrl[2] = 0.2
    for _ in range(100):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        viewer_window.sync()

    target = np.array([0, -1.0, 1.1])

    for _ in range(1000):
        time.sleep(0.01)
        print(data.site_xpos[0])
        position_Q = data.site_xpos[0]
        jacp = np.zeros((3, 3))
        mujoco.mj_jac(model, data, jacp, None, position_Q, 3)

        J_inv = np.linalg.inv(jacp)
        dX = target - position_Q
        dq = J_inv.dot(dX)

        go_to(data.qpos + 0.1 * dq)

        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        viewer_window.sync()


if __name__ == "__main__":
    start_sim()
    target = np.array([0, -1.0, 1.1])
    time.sleep(1)


