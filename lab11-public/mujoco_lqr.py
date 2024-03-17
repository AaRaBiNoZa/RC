import time
import mujoco
from mujoco import viewer
import numpy as np
import control
import imageio

VIEW = False
queue = []
renderer = None

# Constants
GRAVITY = 9.8  # a classic...
CART_MASS = 8.  # kg
POLE_MASS = 6.56  # kg
TOTAL_MASS = CART_MASS + POLE_MASS
POLE_HALF_LENGTH = 0.06 / 2  # half the pole's length in m

def sim_reset():
    global model, data, viewer_window, renderer
    if "viewer_window" in globals():
        viewer_window.close()
    model = mujoco.MjModel.from_xml_path("mujoco_cartpole.xml")
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if VIEW:
        viewer_window = viewer.launch_passive(model, data)


def sim_step(forward):
    global renderer, queue
    data.actuator("slide").ctrl = forward
    step_size = 0.01
    step_start = time.time()
    mujoco.mj_step(model, data)

    if renderer is not None:
        renderer.update_scene(data)
        frame = renderer.render()
        queue.append(frame)
    if VIEW:
        viewer_window.sync()

        time_until_next_step = step_size - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def linearize(): # TODO maybe add some parameters to this function
    # TODO: implement this function
    A = np.array( #TODO: fill in the matrix
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, (TOTAL_MASS * GRAVITY * 4)/(4*TOTAL_MASS - 3 * POLE_MASS * POLE_HALF_LENGTH), 0],
        ]
    )
    B = np.array( #TODO: fill in the matrix
        [
            [0],
            [1/TOTAL_MASS],
            [0],
            [-1/((4/3)*TOTAL_MASS - POLE_HALF_LENGTH * POLE_MASS)],
        ]
    )
    return A, B


def one_run():
    for _ in range(50):
        sim_step(0.03)

    # TODO: maybe you want to perform some precomputation here
    A, B = linearize() #TODO: maybe you need to add some parameters
    Q = np.diag([1e7,1,1,1 ]) #TODO: fill in the matrix (vector)
    R = [1] #TODO: fill in the matrix (one value)
    K = control.lqr(A, B, Q, R)[0]
    print("K", K)
    for _ in range(400):
        angle = data.joint("hinge").qpos[0]
        print("angle", angle)
        angle_vel = data.joint("hinge").qvel[0]
        xpos = data.body("cart").xpos[0]
        xvel = data.body("cart").cvel[3]
        state = np.array([xpos, xvel, angle, angle_vel])
        sim_step(-(K @ state)[0])


def save_video(queue, filename, fps):
    writer = imageio.get_writer(filename, fps=fps)
    for frame in queue:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    sim_reset()
    one_run()
    save_video(queue, "./mujoco-controlled.mp4", 20)
