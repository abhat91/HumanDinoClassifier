import numpy as np
import time
import cv2

class DinoResultsHandler:
    def __init__(self, database):
        self.database = database


    def showImages(self, queryImage, segImage, kpImage):

        # green box
        (_, cnts, _) = cv2.findContours(segImage.copy(),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.drawContours(kpImage, [rect], -1, (0,255,0),2)

        cv2.imshow("Input with Key Points", kpImage)
        time.sleep(0.50)




    def showTexts(self, matchedResults):

        if len(matchedResults) == 0:
            print("no sample are matched to the query !")
            cv2.waitKey(30)

        else:
            for(i, (score, samplePath)) in enumerate(matchedResults):
                description = self.database[samplePath[samplePath.rfind("/") + 1:]]
                print("{}.{:.2f}% : {}".format(i + 1, score * 100, description))

                results = cv2.imread(samplePath) # only show the highest matching image
                cv2.imshow("Right: Matched Sample", results)