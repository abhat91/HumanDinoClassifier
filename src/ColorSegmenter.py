import cv2
import numpy as np

class ColorSegmenter:
    lower_magenta = np.array([125,80,80])
    upper_magenta = np.array([150,255,255])

    @staticmethod
    def getMagentaBlob(bgrimg):
        hsvimg = cv2.cvtColor(bgrimg,cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsvimg,(7,7),0)



        # Threshold the HSV image to get only magenta
        mask = cv2.inRange(blur, ColorSegmenter.lower_magenta, ColorSegmenter.upper_magenta)
        mask = cv2.erode(mask,np.ones((2,2),np.uint8),iterations = 3)
        # mask = cv2.dilate(mask,np.ones((3,3),np.uint8),iterations = 3)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(bgrimg,bgrimg, mask= mask)
        return res
