from __future__ import print_function
from descriptor import Descriptor
from matcher import  Matcher
import argparse
import glob
import csv
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--database", required = True, help = "path to the object information database csv file")
ap.add_argument("-s", "--samples", required = True, help = "path to the sample(training) data folder")
ap.add_argument("-q", "--query", required = True, help = "path to the query image")

args = vars(ap.parse_args())
ap.add_argument("-f", "--sift", type = int, default = 0, help = "use SIFT = 1, not use = 0")

database = {}

for line in csv.reader(open(args["database"])):
    database[line[0]] = line[1:]

useSIFT = args["sift"] > 0
useHamming = args["sift"] == 0
ratio = 0.7
minMatches = 30

if useSIFT:
    minMatches = 50


descriptor = Descriptor(useSIFT = useSIFT)
matcher = Matcher(descriptor, glob.glob(args["samples"] + "/*.png"),
                    ratio = ratio,
                    minMatches = minMatches,
                    useHamming = useHamming)

queryImage = cv2.imread(args["query"])
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = descriptor.describe(gray)

results = matcher.search(queryKps, queryDescs)

cv2.imshow("Query", queryImage)

if len(results) == 0:
    print("no sample are matched to the query !")
    cv2.waitKey(0)

else:
    for(i, (score, samplePath)) in enumerate(results):
        description = database[samplePath[samplePath.rfind("/") + 1:]]
        print("{}.{:.2f}% : {}".format(i + 1, score * 100, description))

        results = cv2.imread(samplePath)
        cv2.imshow("Matched Sample", results)
        cv2.waitKey(0)


