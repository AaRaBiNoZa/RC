import cv2
import os
import numpy as np
import random
from scipy import stats
import matplotlib.pyplot as plt



def find_checkerboard_corners(img):
    found, corners = cv2.findChessboardCorners(img, (8,5))
    print(corners.shape)
    bl_w = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.cornerSubPix(
        bl_w, corners, 
        (11,11), 
        (-1,-1), 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    return found, corners

def find_checkerboard(dirname='data'):
    for fname in os.listdir(dirname):
        f = os.path.join(dirname, fname)

        img = cv2.imread(f)

        found, corners = find_checkerboard_corners(img)

        cv2.drawChessboardCorners(img, (8,5), corners, found)

        # Using cv2.imshow() method 
        # Displaying the image 
        cv2.imshow(fname, img) 
        
        # waits for user to press any key 
        # (this is necessary to avoid Python kernel form crashing) 
        cv2.waitKey(0) 
        
        # closing all open windows 
        cv2.destroyAllWindows() 

def coordinates(point):
    return [int (i) for i in tuple(point.ravel())]

def draw(img, corners, imgpts):
    # WARNING: openCV uses BGR color space
    corner = coordinates(corners[0].ravel())
    img = cv2.line(img, corner, coordinates(imgpts[0]), (0,0,255), 5)
    img = cv2.line(img, corner, coordinates(imgpts[1]), (0,255,0), 5)
    img = cv2.line(img, corner, coordinates(imgpts[2]), (255,0,0), 5)
    return img

def calibrate_cameras(dirname='data', show=True):
    points = get_obj_points()

    images = []
    corners = []

    for fname in os.listdir(dirname):
        f = os.path.join(dirname, fname)

        img = cv2.imread(f)
        images.append(img)

        size = (img.shape[1], img.shape[0])
        found, crnrs = find_checkerboard_corners(img)
        corners.append(crnrs)
    

    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera([points] * len(corners), corners, size, None, None)
    print(camera_matrix)
    print(type(camera_matrix))
    print(dist_coeffs)
    print(type(dist_coeffs))
    print(images[0].shape)

    alpha = 0 # TODO: try 0 and 1
    rect_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, alpha)[0]

    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), rect_camera_matrix, size, cv2.CV_32FC1)

    if show:
        for img, cor in zip(images, corners):
            _, rvec, tvec = cv2.solvePnP(points, cor, camera_matrix, dist_coeffs)
            axis_ends, _ = cv2.projectPoints(np.eye(3,dtype=float) * 100, rvec, tvec, camera_matrix, dist_coeffs)
            
            draw(img, cor, axis_ends)
            

            rect_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

            stacked = np.concatenate((img, rect_img), axis=1)

            # Using cv2.imshow() method 
            # Displaying the image 
            cv2.imshow(fname, stacked) 
            
            # waits for user to press any key 
            # (this is necessary to avoid Python kernel form crashing) 
            cv2.waitKey(0) 
            
            # closing all open windows 
            cv2.destroyAllWindows() 

    return rect_camera_matrix, camera_matrix, dist_coeffs

        

        
def get_obj_points(width=30, num_x=9, num_y=6):
    xs = np.arange(0, 30 * (num_x - 1), 30, dtype=np.float32)
    ys = np.arange(0, 30 * (num_y - 1), 30, dtype=np.float32)
    zs = np.arange(0, 30 * 1, 30, dtype=np.float32)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='xy')

    s = np.stack([xx, yy, zz], axis=-1)

    # xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    # s = np.stack([yy, xx, zz], axis=-1)

    return s.reshape(-1,3)

def value(alpha = 0.95):
    x = 1000 * random.random()
    y = 0.5 * x + random.gauss(0, 3) + 150
    if random.random() < alpha:
        y = random.uniform(150, 650)
    return x, y



def ransac():
    values = [value() for _ in range(1000)]
    values.sort()

    xs = np.array([v[0] for v in values])
    ys = np.array([v[1] for v in values])
    res = stats.linregress(xs, ys) 
    plt.plot(xs, ys, 'o', label='original data')
    plt.plot(xs, res.intercept + res.slope*xs, 'r', label='fitted line')

    best_model = None
    best_count = 0
    for k in range(1000):
        sample = random.sample(values, 2)
        maybe_inliers = set(sample)
        maybe_model = stats.linregress(x=np.array([v[0] for v in sample]), y=np.array([v[1] for v in sample]))
        also_inliers = set()

        for x, y in values:
            y_hat = maybe_model.intercept + maybe_model.slope *  x
            if (y_hat - y) ** 2  < 10:
                also_inliers.add((x,y))
        
        count = len(maybe_inliers | also_inliers)
        if count > best_count:
            best_model = maybe_model
            best_count = count

    plt.plot(xs, best_model.intercept + best_model.slope * xs, 'g', label='RANSAC')
    print(best_count)
    print(best_model.intercept)
    print(best_model.slope)
    plt.legend()
    plt.show()


if __name__ == '__main__' :
    # find_checkerboard()
    # img = cv2.imread('data/img01.jpg')

    # _, crnr = find_checkerboard_corners(img)
    # cv2.circle(img, crnr[0][0].astype(int), 3, (0,0,255), 1)
    # cv2.circle(img, crnr[1][0].astype(int), 3, (0,0,255), 1)
    # print(crnr[0][0])
    # print(crnr[1][0])

    #  # Using cv2.imshow() method 
    # # Displaying the image 
    # cv2.imshow('a', img) 
    
    # # waits for user to press any key 
    # # (this is necessary to avoid Python kernel form crashing) 
    # cv2.waitKey(0) 
    
    # # closing all open windows 
    # cv2.destroyAllWindows() 
    # print(img.shape)
    calibrate_cameras()
    # ransac(10)
