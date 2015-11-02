# HumanDinoClassifier
Based on the human pose, the algorithm picks out the dinosaur (By team Charizard)

# Computer Setup
If you are using a MAC, you may follow this paper to get OpenCV3.0 and Python 2.7
<http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>

# Strategy
Our software will have 4 parts.
1) Canny edge detection of object (assume reasonably solid background, rotation within +/- 10 degrees).
2) Trim extranneous edges. Clean up the object and background edges.
3) Determine optimal image overlaps for each dino. Classify the % the image matches each dino pose.
** Note: when the Dino images themselves go into our system, they should return 99-100% of their own pose.
4) Output HumanDino classification (% trex, % stegasaurus, % triceritops).
