from __future__ import print_function
import numpy as np
import cv2

im = cv2.imread('../testImages/dino5.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)

# CHAIN_APPROX_SIMPLE:  compresses horizontal, vertical, and diagonal segments and leaves only their end points.
# For example, an up-right rectangular contour is encoded with 4 points.
# CHAIN_APPROX_NONE: stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of
# the contour will be either horizontal, \\\vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
# im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


print (contours[0])
# To draw the contours, cv2.drawContours function is used. It can also be used to draw any shape provided you have its
# boundary points. Its first argument is source image, second argument is the contours which should be passed as a
# Python list, third argument is index of contours (useful when drawing individual contour.
# To draw all contours, pass -1) and remaining arguments are color, thickness etc.
cnt = contours[0]
# I find that contours[0] is the largest border of the image and if you run with contours[1] is will be different.
cv2.drawContours(im2, [cnt], -1, (255,255,255), 3)
cv2.imshow("Output", im2)
cv2.waitKey(0)


