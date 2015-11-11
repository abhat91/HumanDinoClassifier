import numpy as np
import cv2

class Descriptor:
    def __init__(self, useSIFT = False):
        self.useSIFT = useSIFT

    def describe(self, image):
        descriptor = cv2.AKAZE_create() #we can use BRISK as well

        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()

        (kps, descs) = descriptor.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps]) #only take the (x,y) attribute and form a NP array.

        return (kps, descs)

