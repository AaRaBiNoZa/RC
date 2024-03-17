import cv2
import numpy as np
from typing import List

def find_color_channel_coords(img, channel):
    mask = img[..., channel] >= 200
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x = (contours[0][...,0].min() + contours[0][...,0].max()) // 2
    y = (contours[0][...,1].min() + contours[0][...,1].max()) // 2
    cv2.drawContours(img, contours, -1, (255,255,255))

    return x,y

def show(
        img_list: List[np.ndarray],
        window_name: str = ''
) -> None:
    """
    Displays each image from input list one by one.
    :param img_list:
    :param window_name:
    :return: None
    """
    s = np.concatenate(img_list, axis=1)

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, s)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('./view.jpg')
    print(find_color_channel_coords(cv2.imread('./view.jpg'), 2))