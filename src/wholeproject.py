
import cv2
import numpy as np
import cv

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


def removebackground(gray, background):
    imgs=cv2.bitwise_not(cv2.bitwise_and(gray, background))
    blobdetecterhelper=BlobDetector()
    erode=blobdetecterhelper.erodeanddilate(imgs, 2,2)
    (thresh, im_bw) = cv2.threshold(erode, 25, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

def preprocessbackground(c, f, avg2):
    t=0
    while(t<100):
        _,f = c.read()
        #Try median
        cv2.accumulateWeighted(f,avg2,0.01)
        res2 = cv2.convertScaleAbs(avg2)
        gray=cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        gray=cv2.medianBlur(gray,5)
        global background
        background=gray
        cv2.imshow('avg2',gray)
        k = cv2.waitKey(20)
        t+=1
        if k == 27:
            break
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
background=0
gray=preprocessbackground(c, f, avg2)
colorblobdetect=BlobDetector()
_,f = c.read()
gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
xsize, ysize= np.shape (gray)

while True:
    _,f = c.read()


    hsvimg = cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsvimg,(7,7),0)
     # define range of blue color in HSV
    lower_magenta = np.array([125,80,80])
    upper_magenta = np.array([150,255,255])



    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(blur, lower_magenta, upper_magenta)
    mask = cv2.erode(mask,np.ones((2,2),np.uint8),iterations = 3)
    # mask = cv2.dilate(mask,np.ones((3,3),np.uint8),iterations = 3)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(f,f, mask= mask)
    # gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    # image_nobackground=removebackground(gray, background)


    # colorblobdetect.getBlob(f)
    # bg=0.4
    # blb=0.6
    # op=cv2.addWeighted(image_nobackground, bg, colorblobdetect.imageBlob, blb, 0)
    # erodedImage=cv2.erode(op, None, iterations = 2)

    # (thresh, im_bw) = cv2.threshold(erodedImage, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # b,g,r = cv2.split(f)
    # nb=np.minimum(im_bw, b)

    # ng=np.minimum(im_bw, g)
    # nr=np.minimum(im_bw, r)
    #new=cv2.merge((nb, ng, nr))

    objectdetection=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(objectdetection, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for csp in cnts:
        print csp
    cv2.imshow('cannyOutput',closed)
    #cv2.imshow('BackgroundRemoved',new)
    k = cv2.waitKey(20)
    if k == 27:
        break

cv2.destroyAllWindows()
c.release()
