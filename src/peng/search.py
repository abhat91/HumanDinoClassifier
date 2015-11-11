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
ap.add_argument("-S", "--sift", type = int, default = 0, help = "use SIFT = 1, not use = 0")

database = {}

for line in csv.reader(open(args["database"])):
    database[line[0]] = line[1:]

