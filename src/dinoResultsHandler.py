import numpy as np
import cv2

class DinoResultsHandler:
    def __init__(self, database):
        self.database = database

    def drawGreenBox(self, queryImage, segImage, kpImage):
        # green box
        gray = cv2.cvtColor(segImage,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,0)
        (_, cnts, _) = cv2.findContours(thresh,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.drawContours(kpImage, [rect], -1, (0,255,0),2)

        return kpImage

    def showTexts(self, matchedResults):

        if len(matchedResults) == 0:
            print("No samples are matched to the query !")
        else:
            for(i, (score, samplePath)) in enumerate(matchedResults):
                description = self.database[samplePath[samplePath.rfind("/") + 1:]]
                print("{}.{:.2f}% : {}".format(i + 1, score * 100, description))

                results = cv2.imread(samplePath) # only show the highest matching image
                cv2.imshow("Right: Matched Sample", results)
                cv2.waitKey(5000)
