from __future__ import print_function
from dinoDescriptor import DinoDescriptor
from dinoMatcher import  DinoMatcher
from dinoSegmenter2 import DinoSegmenter2
import numpy as np
import argparse
import glob
import csv
import cv2

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

inputImage=cv2.imread(args["query"])


segmenter = DinoSegmenter2()
# cv2.imshow('Segmented', segImage)
# cv2.waitKey(0)

descriptor = DinoDescriptor(useSIFT = useSIFT)
dinoMatcher = DinoMatcher(descriptor, glob.glob(args["samples"] + "/*.png"), ratio = ratio, minMatches = minMatches, useHamming = useHamming)

# queryImage = cv2.imread(args["query"])

# capture from webcam
cap = cv2.VideoCapture(0)
while(True):
    ret, inputImage = cap.read()
    segImage = segmenter.segmentImage(inputImage)
    (queryKps, queryDescs) = descriptor.describe(segImage)
    # To  show the key points
    KpImage = cv2.drawKeypoints(inputImage, descriptor.kpsRaw, None)
    # cv2.waitKey(0)
    cv2.imshow("Input with Key Points", KpImage)

    # cv2.waitKey(0)
    # x = raw_input('press n to continue: ')
    # if  x == 'n':
    #     continue
    # else:
    #     break

    results = dinoMatcher.search(queryKps, queryDescs)


    if len(results) == 0:
        print("no sample are matched to the query !")
        cv2.waitKey(30)

    else:
        for(i, (score, samplePath)) in enumerate(results):
            description = db[samplePath[samplePath.rfind("/") + 1:]]
            print("{}.{:.2f}% : {}".format(i + 1, score * 100, description))

            results = cv2.imread(samplePath) # only show the highest matching image
            cv2.imshow("Right: Matched Sample", results)


    # cv2.waitKey(0)
    # x = raw_input('press n to continue: ')
    # if  x == 'n':
    #     continue
    # else:
    #     break


