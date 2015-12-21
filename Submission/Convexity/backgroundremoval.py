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
            # Get average of background with weight of 0.01 for new images
            cv2.accumulateWeighted(f,avg2,0.01)
            t+=1
        #normalize back to 8-bit values
        res2 = cv2.convertScaleAbs(avg2)
        #Remove noise with median blur
        res2=cv2.medianBlur(res2,5)

        return res2

    @staticmethod
    def foregroundMask(color, background):
        b = cv2.split(background)
        im = cv2.split(color)
        mask = None
        out = None
        # For each channel (B-G-R) in the image:
        for i in range(len(b)):
            # Channel c gets a median blur of same size as the kernel
            # used in the preprocessed background
            c = cv2.medianBlur(im[i],5)
            # Gets difference between background and image (chanel-wise)
            imgs=cv2.absdiff(im[i], b[i])
            # Run a Binary OTSU threshold to get what is background and what is
            # not based on the absolute difference.
            (thresh, im_bw) = cv2.threshold(imgs, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Close the Thresholded image to remove
            # small noises in the threshold
            # Reopen the Threshold
            im_bw=cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            im_bw=cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            # Merge Channel Masks
            if mask is None:
                # If it's the first channel being processed
                mask = im_bw
            else:
                # Else, merge channels by running an OR
                # (wherever it's above the threshold, it's an object)
                mask = cv2.bitwise_or(mask,im_bw)
        return mask
