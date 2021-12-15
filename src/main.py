import cv2 as cv
import time
from detection import ObjectDetection
from collect_data import window_capture


cascade_enemy = cv.CascadeClassifier("cascade_files/cascade_10/cascade.xml") # Using model 10 for best results

detect_enemy = ObjectDetection(None)

if __name__ == "__main__":
    loop_time = time.time()
    while True:
        screenshot = window_capture()

        rectangles = cascade_enemy.detectMultiScale(screenshot)

        detection_image = detect_enemy.draw_rectangles(screenshot, rectangles)

        cv.imshow("Matches", detection_image)

        print("FPS: {}".format(1 / (time.time() - loop_time)))
        loop_time = time.time()

        key = cv.waitKey(1)
        if key == ord("q"):
            cv.destroyAllWindows()
            break
