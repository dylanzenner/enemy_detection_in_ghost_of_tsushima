import cv2 as cv
import numpy as np
from mss.windows import MSS as mss
import time


def window_capture():
    """
    Captures the monitor window (area to be specified by user). We are using this in place of a camera because we are
    trying to capture the video on the screen.
    :return: An output image to be used for imshow in the while loop
    """
    with mss() as sct:
        monitor = {"top": 27, "left": 0, "width": 960, "height": 547}
        img = sct.grab(monitor)
        img = np.array(img, dtype=np.uint8)
        np.flip(img[:, :, :3])
    return img


# keep track of how many images have been taken
positive_count = 0
negative_count = 0

loop_time = time.time()
while True:
    screenshot = window_capture()
    cv.imshow("test", screenshot)

    # print('FPS: {}'.format(1 / (time.time() - loop_time))) useful for seeing the FPS
    loop_time = time.time()

    key = cv.waitKey(1)
    if key == ord("q"):
        cv.destroyAllWindows()
        break
    elif key == ord("p"):
        cv.imwrite("positive_images/{}.jpg".format(loop_time), screenshot)
        positive_count += 1
        print("Positive count: {}".format(positive_count))
    elif key == ord("n"):
        cv.imwrite("negative_images/{}.jpg".format(loop_time), screenshot)
        negative_count += 1
        print("Negative count: {}".format(negative_count))
