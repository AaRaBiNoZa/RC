"""
Stub for homework 2
"""
import os
import random
import time
from typing import Tuple, Sequence

import cv2
import mujoco
import numpy as np
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("lab7-public/car.xml")
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)

IM_WIDTH = 640
IM_HEIGHT = 480


def find_color_contours(
        img: np.ndarray,
        color: str,
        name: str = None,
) -> Sequence[np.ndarray]:
    """
    Finds contours of object with a given color.
    :param img: input image in BGR
    :param color: a string from ["red", "green", "blue"]
    :param name: name of a file to save the image with contours to for debug
    purposes. It creates a folder "./images_debug" if didn't exist before. If
    None, no file is saved and no folder is created.
    :return: contours of the object with given colour as returned by
    cv2.findContours
    """

    # cutting the image from below, because the wheels have some green on them
    img = img[:-80, ...]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = None

    if color == "red":
        mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
        mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))

        mask = mask1 | mask2
    elif color == "green":
        mask = cv2.inRange(img_hsv, (45, 50, 20), (75, 255, 255))
    elif color == "blue":
        mask = cv2.inRange(img_hsv, (115, 50, 20), (125, 255, 255))

    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        cv2.imwrite(f'./images_debug/{name}.png', img)
        return contours

    if name:
        if not os.path.isdir('./images_debug'):
            os.mkdir('./images_debug')
        cv2.drawContours(img, contours, -1, (255, 255, 255))
        cv2.imwrite(f'./images_debug/{name}.png', img)

    return contours


def check_if_contours_below(
        contours: Sequence[np.ndarray],
        y_thresh: int
) -> bool:
    """
    Checks if any part of the detected object is below a given y in the input
    image. (Veeery rough estimation of distance)
    :param contours: contours as returned by find_color_contours
    :param y_thresh: threshold y value to test
    :return: True or False if some part of the object was below
    """
    if not contours:
        raise ValueError("Contours must be non empty")

    y_max = contours[0][..., 1].max()

    if y_max >= y_thresh:
        return True

    return False


def rotation_to_x_coord(
        x: int,
        steps: int
) -> float:
    """
    For given x and steps it returns the value of rotation so that after
    steps, the car is approximately heading in that direction.
    IMPORTANT: it is an approximation, but it looks like it roughly fits that
    rotation * steps = 50 is a little under 45 degrees (varies with different
    steps).The formula is derived from the fact, that I assume that if we
    want to head to theleft or right side of the pov, we should aim for a
    little under 45 degrees, if we want to head straight as is - rotation
    should be approximately 0. So rotation is proportional to the distance
    from the middle of the pov.
    :param x: x coord for the car to be headed relative to current pose.
    :param steps: in how many steps should the car head there.
    :return: float for appropriate rotation
    """
    return ((IM_WIDTH // 2 - x) / IM_WIDTH) * (50 / steps)


def get_contours_center(
        contours: Sequence[np.ndarray]
) -> Tuple[int, int]:
    """
    Find center of given contours.
    :param contours: contours as returned by find_color_contours
    :return: x and y coordinates of the center of contours
    """
    x = (contours[0][..., 0].min() + contours[0][..., 0].max()) // 2
    y = (contours[0][..., 1].min() + contours[0][..., 1].max()) // 2

    return x, y


def drive_to_color(
        img: np.ndarray,
        color: str,
        rot_dir: int = 1,
        steps: int = 1000,
        speed: float = 0.1,
        y_thresh: int = 340,
        view: bool = True,
        rotation_thresh: float = 1e-3,
        debug_name: str = None,
        offset: int = 0,
        max_sim_steps: int = 1000,
) -> np.ndarray:
    """
    Drive the car to a given color. Stop if the colour contours are below
    y_thresh
    :param img: input image in RGB
    :param color: color string from ["red", "green", "blue"]
    :param steps: steps in 1 simulation step
    :param speed: speed of the car
    :param y_thresh: threshold to determine that we arrived
    :param rot_dir: direction of rotation (postive = left, negative = right)
    for the times that we don't detect a given color.
    :param view: if should be viewed on the simulator
    :param rotation_thresh: it's a parameter that defines the trajectory of
    the car.
    During the computation we will compute needed rotation and with this
    parameter we can
    determine when should the care drive forward with given rotation and when
    it should first rotate in place to head the right direction.
    :param debug_name: name of the debuf file
    :param offset: we can decide to try to drive the car to a place with some
    offset from the center, although it's tricky. Intuitively, the name might
    suggest that after setting this, we just drive to the left/right of the
    center, but in reality, if we set this parameter, the center shifts, so the
    place we are headed to also kind of changes.
    :param max_sim_steps: max simulation steps to drive without arriving to
    destination
    :return: current image from the car camera (in RGB)
    """
    for _ in range(max_sim_steps):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        contours = find_color_contours(img, color, debug_name)
        if not contours:
            # if didn't detect the color, try to rotate by a little less than
            # 90 degrees
            img = sim_step(0, 100 / steps * rot_dir, steps, view)
        elif check_if_contours_below(contours, y_thresh):
            break
        else:
            x_center, _ = get_contours_center(contours)
            x_center += offset

            rotation = rotation_to_x_coord(x_center, steps=steps)
            if abs(rotation) > rotation_thresh:
                img = sim_step(0, rotation, steps, view)
            else:
                img = sim_step(speed, rotation, steps, view)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def drive_to_color_task3(
        img: np.ndarray,
        color: str,
        max_sim_steps: int,
        steps: int,
        view: bool,
        offset: int,
        minmax: str,
        debug_name: str = None,
) -> np.ndarray:
    """
    Basically the same as the function without _task3(), except that here we
    have a parameter minmax that says if we should instead of center x
    coordinate, aim for left/right edge of the object. (Useful for driving
    around the object)
    :param img:
    :param color:
    :param max_sim_steps:
    :param steps:
    :param view:
    :param debug_name:
    :param y_thresh:
    :param offset:
    :param minmax:
    :return:
    """
    for _ in range(max_sim_steps):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        contours = find_color_contours(img, color, debug_name)
        if not contours:
            # if it loses the sight of the object - if we are going to the
            # left edge, we want to rotate to the right
            # the opposite otherwise
            if minmax == 'min':
                img = task3_step(0, -100 / steps, steps, view)
            else:
                img = task3_step(0, 100 / steps, steps, view)

        elif check_if_contours_below(contours, IM_HEIGHT):
            return img
        else:
            if minmax == 'min':
                x = contours[0][..., 0].min()
            else:
                x = contours[0][..., 0].max()
            x += offset
            rotation = rotation_to_x_coord(x, steps=steps)
            img = task3_step(0.1, rotation, steps, view)

    return img


def go_around(
        view=True
) -> np.ndarray:
    """
    As in the name - go around an object (but end up on the line between
    initial position of car and the object you went around)
    :param view: if should be viewed on the simulator
    :return: image in RGB
    """
    sim_step(0, 0.114 / 2, 1000, view)  # turn 45 degrees left
    sim_step(0.5, 0, 1000, view)  # go straight
    sim_step(0, -0.1111, 1000, view)  # turn 90 degrees right
    return sim_step(0.5, 0, 1000, view)  # go straight


def sim_step(forward, turn, steps=1000, view=False):
    data.actuator("forward").ctrl = forward
    data.actuator("turn").ctrl = turn
    for _ in range(steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / 10)

    renderer.update_scene(data, camera="camera1")
    img = renderer.render()
    return img


def task_1_step(turn):
    return sim_step(0.1, turn, steps=200, view=True)


def task_1():
    steps = random.randint(0, 2000)
    img = sim_step(0, 0.1, steps, view=False)

    # TODO: change the lines below,
    # for car control, you should use task_1_step(turn) function
    # you can change anything below this line
    print(f'Steps: {steps}')
    print(f'Initial ball pos: {data.body("target-ball").xpos}')

    while True:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        contours = find_color_contours(img, 'red')
        if not contours:
            # did not find contours, rotate to find the object
            img = task_1_step(0.5)
        elif check_if_contours_below(contours, 350):
            # arrived close to the ball
            break
        else:
            # head in the colours direction
            x_center, _ = get_contours_center(contours)
            rotation = rotation_to_x_coord(x_center, steps=200)
            img = task_1_step(rotation)

    # rotate some to the left and move (right now car is in front of the ball)
    task_1_step(30 / 200)
    # move forward a bit - get center of the car closer to the ball
    for _ in range(5):
        task_1_step(0)

    print(f'End car pos: {data.body("car").xpos}')
    print(f'End ball pos {data.body("target-ball").xpos}')
    print(
        f'Distance '
        f'{np.linalg.norm(data.body("target-ball").xpos - data.body("car").xpos)}'
    )

    # at the end, your car should be close to the red ball (0.2 distance is
    # fine)
    # data.body("car").xpos) is the position of the car


def task_2():
    sim_step(0.5, 0, 1000, view=True)
    speed = random.uniform(0.3, 0.5)
    turn = random.uniform(-0.2, 0.2)
    img = sim_step(speed, turn, 1000, view=True)
    # TODO: change the lines below,
    # you should use sim_step(forward, turn) function
    # you can change the speed and turn as you want
    # do not change the number of steps (1000)
    print(f'speed: {speed}')
    print(f'turn {turn}')

    initial_box_pos = data.body('target-box-1').xpos.copy()
    view = True
    """ The idea here is simple. We first drive to the green box, and then 
    drive to the red ball (we are approximately on a
    line between the two). We then drive around the ball, turn back and push it.
    After that, we drive back to the green box and place ourselves so we stop
    the ball.
    """
    img = drive_to_color(
        img,
        'green',
        rot_dir=1,
        view=view,
        offset=-80
    )
    img = sim_step(0.2, -0.114 / 2, 1000, view)
    img = drive_to_color(
        img,
        'red',
        rot_dir=-1,
        view=view
    )
    img = go_around(view)
    # this is just trying to aim in the center of the red ball, when we are
    # behind it. Notice that the speed is 0.
    drive_to_color(
        img,
        'red',
        speed=0,
        y_thresh=IM_HEIGHT,
        view=view,
        max_sim_steps=10
    )

    for _ in range(10):
        # push the ball
        img = sim_step(0.1, 0, 1000, view)
    for _ in range(5):
        # wait a second
        img = sim_step(0, 0, 1000, view)

    # turn and move a bit so the car doesn't run into the rolling ball
    # when driving to green box
    img = sim_step(0, 0.114 / 2, 1000, view)
    img = sim_step(0.7, 0, 1000, view)
    img = sim_step(0, -0.114 / 2, 1000, view)
    drive_to_color(
        img,
        'green',
        y_thresh=IM_HEIGHT,
        rot_dir=1,
        offset=-50,
        max_sim_steps=20,
        rotation_thresh=1.1,
        view=view
    )

    # this is placing the car in good position near the box and waiting for the
    # ball to hit
    sim_step(0.1, 0.114 / 2, 1000, view)
    sim_step(0.05, 0, 1000, view)
    sim_step(-0.1, -0.111, 1000, view)
    sim_step(-0.05, 0, 1000, view)


    for _ in range(8):
        sim_step(0, 0., 1000, view)
    for _ in range(10):
        sim_step(0, -0.01, 1000, view)

    print(initial_box_pos)
    print(data.body('target-box-1').xpos)
    box_disp = np.linalg.norm(
        initial_box_pos - data.body("target-box-1").xpos
    )
    box_ball_dist = np.linalg.norm(
        data.body("target-box-1").xpos - data.body("target-ball").xpos
    )

    print(
        f'Box displacement {box_disp}'
    )
    print(
        f'Box distance from ball {box_ball_dist}'
    )
    # at the end, red ball should be close to the green box (0.25 distance is
    # fine)


drift = 0


def task3_step(forward, turn, steps=1000, view=False):
    return sim_step(forward, turn + drift, steps=steps, view=view)


def task_3():
    global drift
    drift = np.random.uniform(-0.1, 0.1)

    # TODO: change the lines below,
    # you should use task3_step(forward, turn, steps) function
    print(f'Drift is {drift}')

    """ The idea is that if I reduce the number of steps that the car makes in
    each simulation step, the drift doesn't matter that much, because I can
    correct the rotation more often.
    """
    steps = 10
    view = True

    # get some vision of what's going on
    img = task3_step(0., 0, view=view, steps=0)

    # drive to the left edge of green box with huge offset, makes the car
    # try to drive aroun the green box on a circle with quite a big radius
    img = drive_to_color_task3(
        img,
        'green',
        max_sim_steps=550,
        steps=steps,
        view=view,
        offset=-300,
        minmax='min',
    )

    # approximately here, we'll be to the left of the green box, so we try to
    # move a bit to the blue box in similar manner - trying to go around it
    # with big radius but from the right
    img = drive_to_color_task3(
        img,
        'blue',
        max_sim_steps=300,
        steps=steps,
        view=view,
        offset=100,
        minmax='max'
    )
    # at the end, car should be between the two boxes

    car_pos = data.body('car').xpos
    dist_from_green = np.linalg.norm(data.body('target-box-1').xpos - car_pos)
    dist_from_blue = np.linalg.norm(data.body('target-box-2').xpos - car_pos)

    print(f'Dist from green {dist_from_green}')
    print(f'Dist from blue {dist_from_blue}')


if __name__ == '__main__':
    task_3()

    viewer.close()
    del renderer
    del viewer