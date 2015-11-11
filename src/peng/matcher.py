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

    # now search the thing
    def search(self, queryKps, queryDescs):
        results = {}

        for dataPath in self.dataPaths:
            obj = cv2.imread(dataPath)
            gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
            (kps, descs) = self.descriptor.describe(gray)

            score = self.match(queryKps, queryDescs, kps, descs)
            results[dataPath] = score

        if len(results) > 0:
            # sort the result for having the most possible match at the first
            results = sorted([(a,b) for (b,a) in results.items() if a > 0], reverse = True)

        return results

    # now match the query obj (B) with our database (A)
    def match(self, kpsA, featuresA, kpsB, featuresB):
        matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append(m[0].trainIdx, m[0].queryIdx)

        if len(matches) > self.minMatches:
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

            return float(status,sum()) / status.size # matching ratio against the object in database

        return -1.0 # no possible match