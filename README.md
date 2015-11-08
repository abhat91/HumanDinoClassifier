# HumanDinoClassifier
Based on the human pose, the algorithm picks out the dinosaur (By team Crocodile)

# Computer Setup
If you are using a MAC, you may follow this paper to get OpenCV3.0 and Python 2.7
<http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>

# Strategy
Our initial software will have 5 parts:
- Object Recognition
  - Input: Original Image
  - Output: X, Y, Width, Height
- Edge Detection
  - Input: Original Image, X, Y, Width, Height
  - Output: Edge Detected Image
- Shape Represention -> Configuration
  - Input: Edge Detected Image
  - Output: Freeman Chain
- Shape Matching against Configuration
  - Input: Freeman Chain
  - Output: Three percentages
- Output
  - Overlay three percentages and some text on the Original Image

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
- findcontours()

# Delegate Work
 - Object Recognition = Aditya
 - Edge Detection = Renato
 - Shape Representation, Configuration = Peng
 - Shape Matching = Joe

 # Future Possibilities
 - Take webcam as input
 - Write OpenCV methods/functions ourselves
