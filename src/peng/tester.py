import cv2
from dinoSegmenter2 import DinoSegmenter2

image = cv2.imread("./queries/query12_Rot_nomark.png")
tester1 = DinoSegmenter2()
tester1.segmentImage(image)
