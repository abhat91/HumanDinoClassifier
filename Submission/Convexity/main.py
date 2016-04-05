
import cv2
import numpy as np
import cv
from ColorSegmenter import ColorSegmenter
from backgroundremoval import BackgroundRemoval
from operator import itemgetter
import copy

def classifyObject(ratioOfAreas):
    if ratioOfAreas>0.5 and ratioOfAreas<0.65:
        return 'T-Rex'
    elif ratioOfAreas>0.70 and ratioOfAreas<0.77:
        return 'Stegosaurus'
    elif ratioOfAreas>0.78 and ratioOfAreas<0.84:
        return 'Triceratops
    elif ratioOfAreas>0.86:
        return 'Volcano'
return ''


def normalized(b,g,r):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    #b,g,r = cv2.split(img)
    b1 = clahe.apply(b)
    g1 = clahe.apply(g)
    r1 = clahe.apply(r)
    #bgr = cv2.merge((b1,g1,r1))
    return b1,g1,r1

c = cv2.VideoCapture(0)
_,f = c.read()
avg2 = np.float32(f)
# Get clean - unmoving background
background=BackgroundRemoval.preprocessbackground(c, f, avg2)
# get new frame
_,f = c.read()
gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
while True:
    _,f = c.read()
    # Remove fixed background from image
    mask=BackgroundRemoval.removebackground(f, background)
    #split channels to remove backround and normalize color
    b,g,r = cv2.split(f)
    # remove background
    nb=np.minimum(mask, b)
    ng=np.minimum(mask, g)
    nr=np.minimum(mask, r)

    # normalize color
    bn,gn,rn = normalized(ng,nr,ng)

    # return to color image
    backgroundRemovedImage=cv2.merge((nb, ng, nr))
    # Color segmentation - Filter out non magenta
    res = ColorSegmenter.getMagentaBlob(backgroundRemovedImage)

    # Canny Edge Detection
    objectdetection=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(objectdetection, 100, 250)

    # Close edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # Get Contours out of edges
    contours,hierarchy = cv2.findContours(closed,2,1)
    areaContours=[]
    res2 = copy.deepcopy(res)
    #For each contour in image:
    for cnt in contours:
        #Only if there are 2 contours or something
        area = cv2.contourArea(cnt)
        #Get Convex hull
        hull = cv2.convexHull(cnt,returnPoints = False)
        if(len(hull)>3 and len(cnt)>3):
            #Get Convexity Defects (where the contour is not convex)
            defects = cv2.convexityDefects(cnt,hull)
            if defects!=None:
                #get contours Areas
                areaContours=areaContours+[(area, len(defects),defects,cnt)]
                if area>1000: # Filter out small objects (noise)
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        cv2.line(res2,start,end,[0,255,0],2) #Draw convex Hull
                    # cv2.fillConvexPoly(res2, np.array([convexpoly]), (255, 255, 255))

    # Now that coonvex hull is drawn on image, obtain area of convex hull as well
    forareaofhull=cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    contoursforhull,hierarchy = cv2.findContours(forareaofhull,2,1)
    areaWithHull=[]
    # for every contours in Hull group
    for cntforhull in contoursforhull:
        #Only if there are 2 contours or something
        area = cv2.contourArea(cntforhull)
        #gets list of convex hulls Areas
        areaWithHull=areaWithHull+[area]

    #Select biggest Area for Hull
    if len(areaContours)>0:
        if max(areaWithHull)>1000:
            #Draw Contour in output image
            defects = max(areaContours,key=itemgetter(1))
            for i in range(defects[2].shape[0]):
                s,e,f,d = defects[2][i,0]
                start = tuple(defects[3][s][0])
                end = tuple(defects[3][e][0])
                cv2.line(res,start,end,[0,255,0],2)
            # Calculate ratio between Hull and shape area
            ratioOfAreas=max(areaContours,key=itemgetter(1))[0]/float(max(areaWithHull))
            # Put on text ratio, Classified Object
            cv2.putText(res,"{:0.2f}".format(ratioOfAreas), (0, res.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(res, classifyObject(ratioOfAreas), (0, res.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    ret, __thresh = cv2.threshold(res, 127, 255,0)
    cv2.imshow('Output',res)
    k = cv2.waitKey(20)
    if k == 27:
        break
cv2.destroyAllWindows()
c.release()
