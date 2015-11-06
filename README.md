# HumanDinoClassifier
Based on the human pose, the algorithm picks out the dinosaur (By team Crocodile)

# Computer Setup
If you are using a MAC, you may follow this paper to get OpenCV3.0 and Python 2.7
<http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>

# Strategy
Our initial software will have 4 parts.
- Object Recognition
- Edge Detection
- Shape Represention -> Configuration
- Shape Matching against Configuration
- Output as % of Dino Match

# OpenCV functions
- cvtcolor()
- inrange()
- getstructuringelement()
- dilate()
- erode()
- ::Params
- ::simpleblobdetector( Params )
- detect()
- canny()

# Delegate Work
 - Object Recognition = Aditya
 - Edge Detection = Renato
 - Shape Representation = Peng
 - Configuration, Shape Matching = Joe
