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
ap.add_argument("-c", "--camera", type = int, default = 1, help = "use camera = 1, use the input query url = 0")


args = vars(ap.parse_args())


db = {}

for line in csv.reader(open(args["db"])):
    db[line[0]] = line[1:]

useSIFT = args["sift"] > 0
useHamming = args["sift"] == 0
ratio = 0.6
minMatches = 15
useCam = args["camera"] > 0

if useSIFT:
    minMatches = 50

descriptor = DinoDescriptor(useSIFT = useSIFT)
dinoMatcher = DinoMatcher(descriptor, glob.glob(args["samples"] + "/*.png"), ratio = ratio, minMatches = minMatches, useHamming = useHamming)
dinoResultsHandler = DinoResultsHandler(db)

if useCam is False:
    queryImage = cv2.imread(args["query"])

# capture from web cam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == True:
        if useCam is True:
            queryImage = utils2.resize(frame, width = 1000)
        # Segment the pink area out
        segImage = ColorSegmenter.getMagentaBlob(queryImage)
        # Describe the query image
        (queryKps, queryDescs) = descriptor.describe(segImage)
        # It is really important to handle the camera idling time.
        if len(queryKps) <= 0:
            print("Please Place The Object In The Camera!")
            cv2.imshow("Query", queryImage)
        # else let's start matching our samples to the query
        else:
            # Matching, the key step
            results = dinoMatcher.search(queryKps, queryDescs)
            # To show the key points on query image
            kpImage = cv2.drawKeypoints(queryImage, descriptor.kpsRaw, None)
            # let's also add the cool green box
            greenBoxImg = dinoResultsHandler.drawGreenBox(queryImage, segImage, kpImage)
            # showing the box must have a timer sleep, otherwise it will be flushed
            cv2.imshow("Green Box", greenBoxImg)
            time.sleep(0.025)
            # print out our matching rates
            dinoResultsHandler.showTexts(results)

        # Nicer function, press 'q' to quite the program
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
        if useCam is False:
            cv2.waitKey(0)
            break

        print(useCam)
    # break when camera is wrong.
    else:
        break
# Release the camera and close all opened windows
cap.release()
cv2.destroyAllWindows()
