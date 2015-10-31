import cv2
import utils


def showImage(frameName, frame):
    cv2.imshow(frameName,frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

dinoImage=utils.Util.getCurrentFolder()+"/../"+"testimages/dino3.png"
frame = cv2.imread(dinoImage,0)
edges = cv2.Canny(frame,150,200)
showImage("Canny Output", edges)
