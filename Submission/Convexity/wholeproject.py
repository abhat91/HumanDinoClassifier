
import cv2
import numpy as np
import cv
from ColorSegmenter import ColorSegmenter
from backgroundremoval import BackgroundRemoval
from operator import itemgetter
import copy

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


def normalized(b,g,r):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    #b,g,r = cv2.split(img)
    b1 = clahe.apply(b)
    g1 = clahe.apply(g)
    r1 = clahe.apply(r)
    #bgr = cv2.merge((b1,g1,r1))
    return b1,g1,r1

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
#cv2.namedWindow('cannyOutput')
################################################################################

c = cv2.VideoCapture(0)
_,f = c.read()
avg2 = np.float32(f)
background=BackgroundRemoval.preprocessbackground(c, f, avg2)
_,f = c.read()
gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
while True:
    _,f = c.read()
    gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    image_nobackground=BackgroundRemoval.removebackground(gray, background)
    b,g,r = cv2.split(f)
    nb=np.minimum(image_nobackground, b)
    ng=np.minimum(image_nobackground, g)
    nr=np.minimum(image_nobackground, r)
    bn,gn,rn = normalized(ng,nr,ng)
    backgroundRemovedImage=cv2.merge((nb, ng, nr))

    res = ColorSegmenter.getMagentaBlob(backgroundRemovedImage)
    objectdetection=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(objectdetection, 100, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours,hierarchy = cv2.findContours(closed,2,1)
    areaContours=[]
    res2 = copy.deepcopy(res)
    for cnt in contours:
        #Only if there are 2 contours or something
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt,returnPoints = False)
        if(len(hull)>3 and len(cnt)>3):
            defects = cv2.convexityDefects(cnt,hull)
            if defects!=None:
                areaContours=areaContours+[(area, len(defects),defects,cnt)]
                if area>1000:
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        cv2.line(res2,start,end,[0,255,0],2)
                    # cv2.fillConvexPoly(res2, np.array([convexpoly]), (255, 255, 255))
    forareaofhull=cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    contoursforhull,hierarchy = cv2.findContours(forareaofhull,2,1)
    areaWithHull=[]
    for cntforhull in contoursforhull:
        #Only if there are 2 contours or something
        area = cv2.contourArea(cntforhull)
        areaWithHull=areaWithHull+[area]
    if len(areaContours)>0:
        if max(areaWithHull)>1000:
            defects = max(areaContours,key=itemgetter(1))
            for i in range(defects[2].shape[0]):
                s,e,f,d = defects[2][i,0]
                start = tuple(defects[3][s][0])
                end = tuple(defects[3][e][0])
                cv2.line(res,start,end,[0,255,0],2)
            ratioOfAreas=max(areaContours,key=itemgetter(1))[0]/float(max(areaWithHull))
            cv2.putText(res,"{:0.2f}".format(ratioOfAreas), (0, res.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            if ratioOfAreas>0.5 and ratioOfAreas<0.65:
                cv2.putText(res,'T-Rex', (0, res.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            elif ratioOfAreas>0.70 and ratioOfAreas<0.77:
                cv2.putText(res, 'Stegosaurus', (0, res.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            elif ratioOfAreas>0.78 and ratioOfAreas<0.84:
                cv2.putText(res, 'Triceratops', (0, res.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            elif ratioOfAreas>0.86:
                cv2.putText(res, 'Volcano', (0, res.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    ret, __thresh = cv2.threshold(res, 127, 255,0)
    cv2.imshow('Output',res)
    k = cv2.waitKey(20)
    if k == 27:
        break
cv2.destroyAllWindows()
c.release()
