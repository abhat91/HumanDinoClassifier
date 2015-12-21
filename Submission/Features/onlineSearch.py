from __future__ import print_function
import cv2
from dinoDescriptor import DinoDescriptor
from dinoMatcher import  DinoMatcher
from ColorSegmenter import  ColorSegmenter
from dinoResultsHandler import DinoResultsHandler
import utils2
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
ratio = 0.7
minMatches = 15

if useSIFT:
    minMatches = 50

descriptor = DinoDescriptor(useSIFT = useSIFT)
dinoMatcher = DinoMatcher(descriptor, glob.glob(args["samples"] + "/*.png"), ratio = ratio, minMatches = minMatches, useHamming = useHamming)
dinoResultsHandler = DinoResultsHandler(db)

# capture from web cam
cap = cv2.VideoCapture(0)
while True:
    ret, queryImage = cap.read()
    if ret == True:
        queryImage = utils2.resize(queryImage, width = 1000)
        # Segment the pink area out
        segImage = ColorSegmenter.getMagentaBlob(queryImage)
        # Describe the query image
        (queryKps, queryDescs, queryKpdRaw) = descriptor.describeQuery(segImage)
        # It is really important to handle the camera idling time.
        if len(queryKps) == 0:
            print("Place The Object In The Camera!")
            cv2.imshow("Query", queryImage)
        # else let's start matching our samples to the query
        else:
            # To show the key points on query image
            kpImage = cv2.drawKeypoints(queryImage, queryKpdRaw, None)
            # let's also add the cool green box
            greenBoxImg = dinoResultsHandler.drawGreenBox(queryImage, segImage, kpImage)
            # showing the box must have a timer sleep, otherwise it will be flushed
            cv2.imshow("Query", greenBoxImg)
            time.sleep(0.025)
            # Matching, the key step
            results = dinoMatcher.search(queryKps, queryDescs)
            # print out our matching rates
            dinoResultsHandler.showTexts(results)

        # Nicer function, press 'q' to quite the program
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    # break when camera is wrong.
    else:
        break
# Release the camera and close all opened windows
cap.release()
cv2.destroyAllWindows()
