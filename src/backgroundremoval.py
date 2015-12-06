import cv2
import numpy as np
import cv

class BackgroundRemoval:
    @staticmethod
    def preprocessbackground(c, f, avg2):
        t=0
        gray=0
        while(t<50):
            _,f = c.read()
            #Try median
            cv2.accumulateWeighted(f,avg2,0.01)
            res2 = cv2.convertScaleAbs(avg2)
            gray=cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
            gray=cv2.medianBlur(gray,5)
            cv2.imshow('avg2',gray)
            k = cv2.waitKey(20)
            t+=1
            if k == 27:
                break
        cv2.destroyWindow('avg2')
        return gray

    @staticmethod
    def removebackground(gray, background):
        imgs=cv2.absdiff(gray, background)
        imgs=cv2.morphologyEx(imgs, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        (thresh, im_bw) = cv2.threshold(imgs, 25, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return im_bw
