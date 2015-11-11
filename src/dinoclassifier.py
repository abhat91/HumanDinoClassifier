import cv2
import utils

frameName = 'Canny Output'
def updateCanny(arg):
    lt = cv2.getTrackbarPos('Edge Linking',frameName)
    ht = cv2.getTrackbarPos('Strong Edges',frameName)
    edges = cv2.Canny(frame,lt,ht)
    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    frame2 = cv2.add(frame,edges)
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
dinoImage=utils.Util.getCurrentFolder()+"/../"+"testimages/dino2.png"
frame = cv2.imread(dinoImage)
frame = cv2.bilateralFilter(frame,9,75,75)
edges = cv2.Canny(frame,0,0)
edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
frame2 = cv2.add(frame,edges)

showImage(frameName, frame2)


