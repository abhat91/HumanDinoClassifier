import cv2
import utils

#Displays the image given the frame name and the frame object
def showImage(frameName, frame):
    """Displays the image given the frame name and the frame object: showImage(frameName, frame)"""
    cv2.imshow(frameName,frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#The main code which has to be extended
dinoImage=utils.Util.getCurrentFolder()+"/../"+"testimages/dino3.png"
frame = cv2.imread(dinoImage,0)
edges = cv2.Canny(frame,150,200)
showImage("Canny Output", edges)
