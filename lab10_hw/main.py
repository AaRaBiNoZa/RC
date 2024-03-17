import argparse
import datetime
import os

import cv2
import numpy as np

process_var = 1  # Process noise variance
measurement_var = 1e4  # Measurement noise variance


class KalmanFilter:
    def __init__(self, process_var, measurement_var_x, measurement_var_y):
        # process_var: process variance, represents uncertainty in the model
        # measurement_var: measurement variance, represents measurement noise

        ### TODO
        ### Change the model to constant acceleration model

        # Measurement Matrix
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
        )

        # Process Covariance Matrix
        self.Q = np.eye(6) * process_var

        # Measurement Covariance Matrix
        self.R = np.array(
            [
                [measurement_var_x, 0],
                [0, measurement_var_y],
            ]
        )

        # Initial State Covariance Matrix
        self.P = np.eye(6)

        # Initial State
        self.x = np.zeros(6)

    def predict(self, dt):
        # State Transition Matrix
        A = np.array(
            [
                [1, 0, dt, 0, 0, 0],
                [0, 1, 0, dt, 0, 0],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Predict the next state
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q
        print(f"Predicted State: {self.x}")

    def update(self, measurement):
        # Update the state with the new measurement
        print(f"Measurement: {measurement}")
        y = measurement - self.H @ self.x
        print(f"y: {y}")
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


def draw_uncertainty(kf, img):
    ### TODO
    ### Draw uncertainty
    std_x = np.sqrt(kf.P[0, 0])
    std_y = np.sqrt(kf.P[1, 1])

    # for 95% prediction interval (0.95^2 ~ 0.9) of normal distribution with given mean and variance
    z = 1.959963984540

    x_range = (kf.x[0] - z * std_x, kf.x[0] + z * std_x)
    y_range = (kf.x[1] - z * std_y, kf.x[1] + z * std_y)

    cv2.rectangle(
        img,
        (int(x_range[0]), int(y_range[0])),
        (int(x_range[1]), int(y_range[1])),
        color=(0, 255, 0),
        thickness=3,
    )


class ClickReader:
    def __init__(self, process_var, measurement_var, window_name="Click Window"):
        self.window_name = window_name
        self.cur_time = datetime.datetime.now()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.kf = KalmanFilter(process_var, measurement_var, measurement_var)

        self.img = 255 * np.ones((500, 500, 3), np.uint8)

    def mouse_callback(self, event, x, y, flags, param):
        # Check if the event is a left button click
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Time: {current_time}, Position: ({x}, {y})")
            new_time = datetime.datetime.now()
            self.kf.predict((new_time - self.cur_time).total_seconds())
            self.cur_time = new_time

            cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)  # Red color, filled circle
            self.kf.update(np.array((x, y)))
            print(f"Updated State: {self.kf.x}")

    def run(self):
        # Main loop to display the window
        while True:
            new_time = datetime.datetime.now()
            self.kf.predict((new_time - self.cur_time).total_seconds())
            self.cur_time = new_time

            cv2.circle(
                self.img, (int(self.kf.x[0]), int(self.kf.x[1])), 2, (255, 0, 0), -1
            )  # Blue color, filled circle

            img_draw = self.img.copy()

            draw_uncertainty(self.kf, img_draw)

            cv2.imshow(self.window_name, img_draw)

            # Exit on pressing the 'ESC' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


class PredefinedClickReader:
    def __init__(
            self,
            process_var,
            measurement_var_x,
            measurement_var_y,
            window_name="Click Window",
    ):
        self.window_name = window_name
        self.cur_time = datetime.datetime.now()
        cv2.namedWindow(self.window_name)
        self.kf = KalmanFilter(process_var, measurement_var_x, measurement_var_y)

        self.img = 255 * np.ones((500, 500, 3), np.uint8)

    def run(self, observation_generator):
        for dt, observation in observation_generator:
            self.kf.predict(dt)
            if observation is not None:
                self.kf.update(observation)
                cv2.circle(
                    self.img,
                    (int(observation[0]), int(observation[1])),
                    2,
                    (0, 0, 255),
                    -1,
                )
            cv2.circle(
                self.img, (int(self.kf.x[0]), int(self.kf.x[1])), 2, (255, 0, 0), -1
            )
            img_draw = self.img.copy()
            draw_uncertainty(self.kf, img_draw)
            cv2.imshow(self.window_name, img_draw)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parabola_generator():
    for x in range(0, 500, 1):
        if np.random.rand(1)[0] > 0.5:
            yield 1, None
        else:
            yield 1, np.array(
                [
                    x + np.random.randn(1)[0] * np.sqrt(1e2),
                    x * (500 - x) / 250 + np.random.randn(1)[0] * np.sqrt(4e2),
                ]
            )


def find_ball_center(
        img,
        name=None
):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red hsv range
    mask1 = cv2.inRange(img_hsv, (0, 50, 50), (5, 255, 255))
    mask2 = cv2.inRange(img_hsv, (175, 50, 50), (180, 255, 255))

    mask = mask1 | mask2

    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # find contours with max area (mask isn't perfect, it catches some blobs from background)
        c = max(contours, key=cv2.contourArea)

        # ball contours' area is in thousands, while outliers are at most couple of hundred
        if cv2.contourArea(c) < 1e3:
            return None
        x_center = (c[..., 0].min() + c[..., 0].max()) / 2
        y_center = (c[..., 1].min() + c[..., 1].max()) / 2

        # for debug purposes
        if name:
            if not os.path.isdir('./images_debug'):
                os.mkdir('./images_debug')
            cv2.drawContours(img, (c,), -1, (255, 255, 255))
            cv2.imwrite(f'./images_debug/{name}.png', img)

        return int(x_center), int(y_center), c
    else:
        return None


class VideoReader:
    def __init__(
            self,
            process_var,
            measurement_var,
            video_path,
            window_name="Video Window",
            fps=29.97,
    ):
        self.video = cv2.VideoCapture(video_path)
        self.kf = KalmanFilter(process_var, measurement_var, measurement_var)
        self.window_name = window_name
        cv2.namedWindow(window_name)
        self.fps = fps

    def run(self):
        ### TODO
        ### Set initial position using the first frame
        ret, frame = self.video.read()
        x, y, ctrs = find_ball_center(frame)

        ball_radius = int(max(ctrs[..., 0].max() - ctrs[..., 0].min(), ctrs[..., 1].max() - ctrs[..., 1].min()) / 2)
        self.kf.x[0] = x
        self.kf.x[1] = y

        # timestep is in a way arbitrary to some extent (controls scale in a way)
        dt = 0.3  # 0.2 also works quite well

        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            ### TODO
            ### Find the red ball in the frame
            ### Use Kalman Filter to track the ball and predict its position

            self.kf.predict(dt)
            observation = find_ball_center(frame)
            if observation is not None:
                self.kf.update(np.array(observation[:2]))
            else:
                cv2.circle(
                    frame, (int(self.kf.x[0]), int(self.kf.x[1])), ball_radius, (0, 0, 255), -1
                )

            # debug
            # cv2.circle(
            #     frame, (int(self.kf.x[0]), int(self.kf.x[1])), ball_radius, (0, 0, 255), -1
            # )

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # add an argument to decide between click, predefined and video
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="click",
        choices=["click", "predefined", "video"],
        help="Mode to run the program in. Options are click, predefined, video",
    )
    args = parser.parse_args()
    if args.mode == "click":
        click_reader = ClickReader(process_var, measurement_var)
        click_reader.run()
    elif args.mode == "predefined":
        ### TODO
        ### Read parabola_generator and set measurement_var_x and measurement_var_y

        predefinedclicker = PredefinedClickReader(0, measurement_var_x=1e2, measurement_var_y=4e2)
        predefinedclicker.run(parabola_generator())
    else:
        assert args.mode == "video"
        video_reader = VideoReader(10, 10, "line.mp4")
        video_reader.run()
