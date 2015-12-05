import cv2
import numpy as np

class DinoSegmenter2:
    def __init__(self):
        #  red, blue, yellow, and gray colors range with upper and lower bound in the image.
        # color = [ B,G,R]
        self.boundaries = [
            ([125,80,80], [150,255,255])]



    def segmentImage(self, image):
         # loop over the boundaries
        for (lower, upper) in self.boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask = mask)

            # return the color segmented image
            return output

    def test(self, image):
         # loop over the boundaries
        for (lower, upper) in self.boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask = mask)

            # show the images
            cv2.imshow("images", np.hstack([image, output]))
            cv2.waitKey(0)





#An example on how to use this class. Remove this when implemented in the main code
