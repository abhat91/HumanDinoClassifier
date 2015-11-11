import numpy as np
import cv2

class Matcher:
    def __int__(self, descriptor, dataPaths, ratio = 0.7,
                minMatches = 30, useHamming = True):
        self.descriptor = descriptor
        self.dataPaths = dataPaths
        self.ratio = ratio
        self.minMatches = minMatches
        self.distanceMethod = "BruteForce"

        if useHamming:
            self.distanceMethod = "-Hamming"