import numpy as np
import cv2

class DinoDescriptor:
    def __init__(self, useSIFT = False):
        self.useSIFT = useSIFT

    def describe(self, image):
        descriptor = cv2.BRISK_create() #we can use BRISK as well

        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()

        (kps, descs) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps]) #only take the (x,y) attribute and form a NP array.

        return (kps, descs)

