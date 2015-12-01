# HumanDinoClassifier by Team Crocodile

## Project Introduction
The output we expect is that our program will tell us who is in the image and which dinosaur is in the image with a query image that contains no more than one person and one dinosaur.

## Project Classes
The classes listed here are not limited to only one big class. They are representing some system boudoirs that have a clear cut. You should considering separate your portion of code into function blocks.

We follow the **MVC design rule** and in this special CV program, our model will be images(trading and query), our controller will be classifiers, out view with be output handing class.

##### Model
1. Image taker
	2. Have to output query image in **PNG** **PNG** **PNG** format!
2. Image handler(if needed)

##### View
1. Results printer
	2. should outputs “In this query image (No person/Peng/Aditya/Renato/Joe) is holding a (dinosaur/T-Rax1/T-Rax2/Fat-dino/Volcano)!
2. Contour highlighter(phase 2)
	3. Highlight the contour of human and dinosaur.
	
##### Controller
1. Human classifier
	2. input: a query inmage
	3. output: (No person/Peng/Aditya/Renato/Joe) with in (running time) and (matching rate)%
2. Dinosaur classifier
	3. input: a query inmage
	3. output: (dinosaur/T-Rax1/T-Rax2/Fat-dino/Volcano) with in (running time) and (matching rate)% 

## Project Evaluation Criteria:
- 20% Soundness of approach
	- make sure the program not crashed 9/10 times.
- 15% Justification
	- make sure the program puts correct names on the queries.
- 20% Analysis
	- make sure the people understands your code
- 15% Testing & examples
	- throw in queries with scaling, rotation and 
- 15% Documentation
	- log all your works
- 10% Presentation
	- present order TBD
- 5% Difficulty
	- add complex BKG queries
- Extra (up to 10%)
	- recognize dinosaurs along with human who is hold it
	


-------------------------------------------------------- 

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
   
  
# Strategy 2
Since the chain code is not invariant on scaling, we will try two solutions: Blobdetect and Keypoint&Feature Vector detect.
At this time, we simplifiy the problem set to "Detect the objects, which we already have in the database, in a query image and show the reasults with highlighted contours of the objects."

- Blob Detection uses the color channel of our dinosaours and we query the program with a simple background image.
- Keypoint&Feature detection is more robust with complicated background and could work on humans as well. 

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
 
Temp work:
- Blob detection - Aditya
- Keypoint&Feature detection - Peng
 

# Future Possibilities
 - Take webcam as input
 - Write OpenCV methods/functions ourselves
 
# Computer Setup
If you are using a MAC, you may follow this paper to get OpenCV3.0 and Python 2.7
<http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>
