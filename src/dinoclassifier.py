import cv2
import utils
import copy
import numpy as np

frameName = 'Canny Output'
def updateCanny(arg):
    lt = cv2.getTrackbarPos('Edge Linking',frameName)
    ht = cv2.getTrackbarPos('Strong Edges',frameName)
    edges = cv2.Canny(frame,lt,ht)
    frame2 = copy.deepcopy(frame)
    contours, _ = cv2.findContours(edges,1,2)
    poly = []
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
                    if i == 0:
                        poly.append(start)
                    poly.append(end)
                    #cv2.line(frame2,start,end,[0,255,0],2)
                    #cv2.circle(frame2,far,5,[0,0,255],-1)
    #print poly
    cv2.fillConvexPoly(frame2,np.array([poly]),(255,255,255))#-1,1,[255,255,255],-1)
    #cv2.drawContours(frame2,contours,-1,[255,0,0],3)
    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    frame2 = cv2.add(frame2,edges)

    cv2.imshow(frameName,frame2)
#Displays the image given the frame name and the frame object
def showImage(frameName, frame):
    """Displays the image given the frame name and the frame object: showImage(frameName, frame)"""
    cv2.namedWindow(frameName)
    cv2.createTrackbar('Edge Linking',frameName,0,1000,updateCanny)
    cv2.createTrackbar('Strong Edges',frameName,0,1000,updateCanny)
    cv2.imshow(frameName,frame)
    cv2.waitKey(0)
    lt = cv2.getTrackbarPos('Edge Linking',frameName)
    ht = cv2.getTrackbarPos('Strong Edges',frameName)
    print lt,ht
    cv2.destroyAllWindows()

#The main code which has to be extended
dinoImage=utils.Util.getCurrentFolder()+"/../"+"testimages/dino4.png"
frame = cv2.imread(dinoImage)
frame = cv2.bilateralFilter(frame,9,75,75)
edges = cv2.Canny(frame,0,0)
# frame2 = copy.deepcopy(frame)
# cv2.drawContours(frame2,countours,0,(0,255,0),2)
edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
frame2 = cv2.add(frame,edges)

showImage(frameName, frame2)
