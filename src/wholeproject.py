
import cv2
import numpy as np
import cv
from ColorSegmenter import ColorSegmenter
from backgroundremoval import BackgroundRemoval

class BlobDetector:
    frame=0
    lower_pink_hue_range=0
    tempimage=0
    imageBlob=0

    #Adjust these for the right values (adjust these again for realtime data)
    boundaries = [[124,10,10], [185,255,255]]

    thresh_lower = np.array(boundaries[0][0], dtype = "uint8")
    thresh_upper = np.array(boundaries[1][0], dtype = "uint8")


    def erodeanddilate(self, image, erosionCount, dilationCount):
        erode = cv2.erode(image, None, iterations = erosionCount)
        dilate = cv2.dilate(erode,None,iterations = dilationCount)
        return dilate

    def blobDetect(self):
        self.imageBlob=self.tempimage


    def thresholderodeanddilate(self):
        #Do this again to get the upper threshold (Double threshold is the norm)
        self.lower_pink_hue_range=cv2.inRange(self.frame, self.thresh_lower, self.thresh_upper)
        self.tempimage=self.erodeanddilate(self.lower_pink_hue_range, 2, 10)


    def getBlob(self, image):
        self.frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.thresholderodeanddilate()
        self.blobDetect()


def normalized(img):
    return img

def getBlob(image):
    blob_detect=BlobDetector()
    blob_detect.getBlob(image)
    return blob_detect.imageBlob

def invertImage(image):
    return cv2.bitwise_not(image)

def getInvertedBlob(image):
    blobImage=getBlob(image)
    return invertImage(blobImage)


def preprocessing(image):
    norm_image=normalized(image)
    inverted_image=getInvertedBlob(norm_image)
    return inverted_image


def nothing(x):
    pass
################################################################################
#Preprocessing stuff
cv2.namedWindow('BackgroundRemoved')
cv2.namedWindow('cannyOutput')
################################################################################

c = cv2.VideoCapture(0)
_,f = c.read()
avg2 = np.float32(f)
background=BackgroundRemoval.preprocessbackground(c, f, avg2)
colorblobdetect=BlobDetector()
_,f = c.read()
gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
sampleImage=cv2.imread('/home/adi/Desktop/HumanDinoClassifier/testimages/dino2.png',0)
sampleimageedges = cv2.Canny(sampleImage, 10, 250)
sampleImageT=cv2.imread('/home/adi/Desktop/HumanDinoClassifier/testimages/dino3.png',0)
sampleimageedgesT = cv2.Canny(sampleImage, 10, 250)
(cnts, _) = cv2.findContours(sampleimageedgesT.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

while True:
    _,f = c.read()
    gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    image_nobackground=BackgroundRemoval.removebackground(gray, background)
    b,g,r = cv2.split(f)
    nb=np.minimum(image_nobackground, b)
    ng=np.minimum(image_nobackground, g)
    nr=np.minimum(image_nobackground, r)
    backgroundRemovedImage=cv2.merge((nb, ng, nr))

    res = ColorSegmenter.getMagentaBlob(backgroundRemovedImage)
    cv2.imshow('RemovedBackground', res)
    objectdetection=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(objectdetection, 100, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours,hierarchy = cv2.findContours(closed,2,1)
    for cnt in contours:
        hull = cv2.convexHull(cnt,returnPoints = False)
        if(len(hull)>3 and len(cnt)>3):
            defects = cv2.convexityDefects(cnt,hull)
            if defects!=None:
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    cv2.line(res,start,end,[0,255,0],2)
                    cv2.circle(closed,far,5,[0,0,255],-1)

    cv2.imshow('cannyOutput',res)
    k = cv2.waitKey(20)
    if k == 27:
        break

cv2.destroyAllWindows()
c.release()
