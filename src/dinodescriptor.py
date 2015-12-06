import numpy as np
import cv2

class DinoDescriptor:
    def __init__(self, useSIFT = False):
        self.useSIFT = useSIFT
        self.kpsRaw = 0
        self.gray = 0

    def describe(self, image):
        # descriptor = cv2.BRISK_create() #we can use BRISK as well
        descriptor = cv2.AKAZE_create() #we can use BRISK as well

        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()
        # make it gray scaled for 2D features
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (self.kps, descs) = descriptor.detectAndCompute(self.gray, None)
        kps = np.float32([kp.pt for kp in self.kps]) #only take the (x,y) attribute and form a NP array.

        return (kps, descs)

    def describeQuery(self, image):
        # descriptor = cv2.BRISK_create() #we can use BRISK as well
        descriptor = cv2.AKAZE_create() #we can use BRISK as well

        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()
        # make it gray scaled for 2D features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kpsRaw, descs) = descriptor.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kpsRaw]) #only take the (x,y) attribute and form a NP array.

        return (kps, descs, kpsRaw)
