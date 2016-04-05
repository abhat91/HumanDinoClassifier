# HumanDinoClassifier by Team Crocodile

## Project Introduction
In this project, we are taking frames from a web-cam and try to classify which dinosaur is in the image(if it contains one). We implement two methods to deal with the classification, one is key point detection and the other one is using convexity method. 

This project is finished and released on Dec 21, 2015.

## How To Run
- `run2.sh` will show a quick demo using key point detection. (You need openCV 3.0 to run this)
- `onlineSearch.py` with command line argument will run the program by using A-KAZE key point method. (You need openCV 3.0 to run this)
- `wholeproject.py` will run the program by using convexity method to classify the object.

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

# Computer Setup
If you are using a MAC, you may follow this paper to get OpenCV3.0 and Python 2.7
<http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>
