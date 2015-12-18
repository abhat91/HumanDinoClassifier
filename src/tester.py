from __future__ import print_function
import cv2
from dinoDescriptor import DinoDescriptor
from dinoMatcher import  DinoMatcher
from ColorSegmenter import  ColorSegmenter
from dinoResultsHandler import DinoResultsHandler
import numpy as np
import argparse
import glob
import csv
import time


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True, help = "path to the object information database csv file")
ap.add_argument("-s", "--samples", required = True, help = "path to the sample(training) data folder")
ap.add_argument("-q", "--query", required = True, help = "path to the query image")
ap.add_argument("-f", "--sift", type = int, default = 0, help = "use SIFT = 1, not use = 0")

args = vars(ap.parse_args())


db = {}

for line in csv.reader(open(args["db"])):
    db[line[0]] = line[1:]

useSIFT = args["sift"] > 0
useHamming = args["sift"] == 0
ratio = 0.6
minMatches = 15

if useSIFT:
    minMatches = 50

# inputImage=cv2.imread(args["query"])


# segmenter = DinoSegmenter2()
# cv2.imshow('Segmented', segImage)
# cv2.waitKey(0)

descriptor = DinoDescriptor(useSIFT = useSIFT)
dinoMatcher = DinoMatcher(descriptor, glob.glob(args["samples"] + "/*.png"), ratio = ratio, minMatches = minMatches, useHamming = useHamming)
dinoResultsHandler = DinoResultsHandler(db)

queryImage = cv2.imread(args["query"])


# Segment the pink area out
segImage = ColorSegmenter.getMagentaBlob(queryImage)

(queryKps, queryDescs) = descriptor.describe(segImage)
# It is really important to handle the camera idling time.
if len(queryKps) == 0:
    print("no key points detected in query!")
    cv2.waitKey(500)
else:
    # To  show the key points
    kpImage = cv2.drawKeypoints(queryImage, descriptor.kpsRaw, None)
    # cv2.waitKey(0)

    results = dinoMatcher.search(queryKps, queryDescs)

    greenBoxImg = dinoResultsHandler.drawGreenBox(queryImage, segImage, kpImage)

    cv2.imshow("Green Box",greenBoxImg)
    time.sleep(0.025)
    cv2.waitKey(0)

    dinoResultsHandler.showTexts(results)

    # if len(results) > 0:
    #     dinoResultsHandler.showImages(queryImage, segImage, kpImage)
    #     dinoResultsHandler.showTexts(results)
    # else:
    #     cv2.imshow("Web Camera",queryImage)

cv2.destroyAllWindows()







