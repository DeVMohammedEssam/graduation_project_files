# importing modulesc
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from modules.func import find_good_matches, draw_on_match


# reading images in gray scale
img1 = cv.imread("lucky_charms.jpg", 0)
img2 = cv.imread('many_cereals.jpg', 0)

# finding matches between the two imported images
good, kp1, kp2 = find_good_matches(img1, img2)
# drawing the detected match
output = draw_on_match(good, img1, kp1, img2, kp2)


# showing the result
cv.imshow("image", output)
cv.waitKey(0)
