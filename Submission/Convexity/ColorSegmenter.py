import cv2
import numpy as np

class ColorSegmenter:
    #Thresholds for Magenta blob
    lower_magenta = np.array([125,30,30])
    upper_magenta = np.array([160,255,255])

    @staticmethod
    def getMagentaBlob(bgrimg):
        #Convert image space to HSV
        hsvimg = cv2.cvtColor(bgrimg,cv2.COLOR_BGR2HSV)
        # Apply Bilateral Filter to remove noise
        blur = cv2.bilateralFilter(hsvimg,9,75,75)
        # Threshold the HSV image to get only magenta
        mask = cv2.inRange(blur, ColorSegmenter.lower_magenta, ColorSegmenter.upper_magenta)
        #Clean residual noise eroding, opening and closing image
        mask = cv2.erode(mask,np.ones((1,1),np.uint8),iterations = 5)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((4,4),np.uint8))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))

        # Apply mask to original image
        res = cv2.bitwise_and(bgrimg,bgrimg, mask= mask)
        return res
