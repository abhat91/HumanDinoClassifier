import cv2
import numpy as np

class BlobDetector:
    frame=0
    lower_pink_hue_range=0
    tempimage=0
    imageBlob=0

    #Adjust these for the right values (adjust these again for realtime data)
    boundaries = [([50,50,50], [255,255,255]), ([100,100,100], [120,255,255])]

    lowthresh_lower = np.array(boundaries[0][0], dtype = "uint8")
    lowthresh_upper = np.array(boundaries[0][1], dtype = "uint8")

    highthresh_lower = np.array(boundaries[1][0], dtype = "uint8")
    highthresh_upper = np.array(boundaries[1][1], dtype = "uint8")


    def erodeanddilate(self, image, erosionCount, dilationCount):
        erode = cv2.erode(image, None, iterations = erosionCount)
        dilate = cv2.dilate(erode,None,iterations = dilationCount)
        return dilate

    def blobDetect(self):
        #TODO:To be done
        self.imageBlob=self.tempimage


    def thresholderodeanddilate(self):
        #Do this again to get the upper threshold (Double threshold is the norm)
        self.lower_pink_hue_range=cv2.inRange(self.frame, self.lowthresh_lower, self.lowthresh_upper)
        self.tempimage=self.erodeanddilate(self.lower_pink_hue_range, 2, 10)


    def getBlob(self, image):
        self.frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.thresholderodeanddilate()
        self.blobDetect()

#An example on how to use this class. Remove this when implemented in the main code
bdetect=BlobDetector()
inputImage=cv2.imread('/home/aditya/Projects/HumanDinoClassifier/testimages/dino5.png')
bdetect.getBlob(inputImage)
cv2.imshow(bdetect.imageBlob)
cv2.waitKey(0)
