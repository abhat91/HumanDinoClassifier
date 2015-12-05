import cv2
from ColorSegmenter import ColorSegmenter
from dinodescriptor import DinoDescriptor

# image = cv2.imread("./queries/query12_Rot_nomark.png")
cap = cv2.VideoCapture(0)
while(True):
    ret, image = cap.read()

    segImage = ColorSegmenter.getMagentaBlob(image)
    cv2.imshow("seg", segImage)
    gray = cv2.cvtColor(segImage, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    descriptor = DinoDescriptor()
    (queryKps, queryDescs) = descriptor.describe(gray)
    KpImage = cv2.drawKeypoints(gray, descriptor.kpsRaw, None)
    cv2.imshow("Input with Key Points", KpImage)

    # cv2.waitKey(100)

# cap = cv2.VideoCapture(0)
# while(True):
#     ret, inputImage = cap.read()
#     segImage = ColorSegmenter.getMagentaBlob(inputImage)
#     gray = cv2.cvtColor(segImage, cv2.COLOR_BGR2GRAY)