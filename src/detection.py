import cv2 as cv
import numpy as np


class ObjectDetection:

    enemy_img = None
    enemy_w = 0
    enemy_h = 0
    method = None

    def __init__(self, enemy_img_path, method=cv.TM_CCOEFF_NORMED):
        if enemy_img_path:
            self.enemy_img = cv.imread(enemy_img_path, cv.IMREAD_UNCHANGED)

            self.enemy_w = self.enemy_img.shape[1]
            self.enemy_h = self.enemy_img.shape[0]

        self.method = method

    def find_enemies(self, zone_img, threshold=0.99, max_results=15):
        """
        Locates the enemy locations on the zone_img and determines the bounding box locations using the matchTemplate
        algorithm
        :param zone_img: The picture containing the enemies we are trying to detect
        :param threshold: set to 0.99 because we are looking for the enemies in zone_img which have the highest matching pixels
        :param max_results: set to 15 so we only find 15 enemies at a time. If we find more than 15 we will only take the locations of 15
        :return: a list of enemy locations each in a rectangle form
        """
        result = cv.matchTemplate(zone_img, self.enemy_img, self.method)

        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        if not locations:
            return np.array([], dtype=np.int32).reshape(0, 4)

        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.enemy_w, self.enemy_h]
            # Add every box to the list twice in order to retain single (non-overlapping) boxes
            rectangles.append(rect)
            rectangles.append(rect)

        rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

        if len(rectangles) > max_results:
            rectangles = rectangles[:max_results]

        return rectangles

    def draw_rectangles(self, zone_img, rectangles):
        """
        Takes in the rectangle locations obtained from find() and draws those rectangles on the zone_img which
        contains a full in game view of the player and surrounding area
        :param zone_img: The image which contains the enemies we are trying to detect
        :param rectangles: the locations of the detected enemies
        :return: zone_img with rectangles drawn on
        """
        line_color = (0, 255, 0)
        line_type = cv.LINE_4

        for (x, y, w, h) in rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box
            cv.rectangle(
                zone_img, top_left, bottom_right, line_color, lineType=line_type
            )

        return zone_img
