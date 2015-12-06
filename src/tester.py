import cv2
import csv
from dinoDescriptor import DinoDescriptor
from dinoMatcher import  DinoMatcher
from ColorSegmenter import  ColorSegmenter
from dinoResultsHandler import DinoResultsHandler

queryImage = cv2.imread("./queries/query12_Rot_nomark.png")

db = {}

for line in csv.reader(open("./dinoDB.csv")):
    db[line[0]] = line[1:]

descriptor = DinoDescriptor()
dinoResultsHandler = DinoResultsHandler(db)


segImage = ColorSegmenter.getMagentaBlob(queryImage)
# gray = cv2.cvtColor(segImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = descriptor.describe(segImage)

kpImage = cv2.drawKeypoints(queryImage.copy(), descriptor.kpsRaw, None)


dinoResultsHandler.showImages(queryImage, segImage, kpImage)

cv2.waitKey(0)


