# WandaVision

## Introduction

In this project, we use the picamera module and a Raspberri Pi to do multiple things. First, we calibrate said camera using a chessboard. Then, we use the camera to detect figures, like polygons, circles, and lines (with their color too). We also implement a password system that uses the camera to detect a certain pattern of figures. Lastly, we use the camera to detect an aruco code (that a person can put in their hand) and follow it, to identify patterns (specifically a circle, a heart, an infinity symbol, and a lightning).

## Methodology

### Calibration

To calibrate the camera, we use a chessboard. We take multiple pictures of the chessboard in different positions and orientations using the function `take_chessboard_images`. Then, we use the pictures to calibrate the camera. We use the function `get_calibration_points` to find the corners of the chessboard in the pictures. Then, we use the function `cv2.calibrateCamera()` to calibrate the camera.

<div align="center">
    <img src="/images/report/calibration.png" alt="WandaVision" width="50%" height="50%">
</div>

### Pattern detection

We are able to detect any n-sided polygon of any color. For that we only need to precompute the exact rgb color of the polygon in the camera. We do that using the file `src/detect_patterns/detect`.   

Then, to detect the figure, we first blur the image using gaussian blur to decrease noise. Then, we segment the image by the color of the polygon (using euclidean distance to the color and setting a threshold), and we binarize by that filter. Next, we want to get rid of any small blobs of the color, so we erode the image.

Now that we have a binarized image with just the shape we want to detect, the next step is to find how many sides it has. We do that by getting the contours of the shape by using `cv2.findContours()`. Then, we approximate the contours using `cv2.approxPolyDP()`, which returns the number of sides of the shape. If we have done everything correctly, we now know how many sides the polygon of a certain color has, so it is detected.

### Sequence decoder

We create a loop that, using the pattern detection algorithm defined before, lets you define a password and keeps asking you to enter it until you do so correctly. 

In order to detect more confidently when a figure enters or disappears, we add the detected figures to a queue for a defined number of frames. We only detect a positive edge when the figure is filling the whole queue, and a negative edge when only Nones are filling it after a figure has been detected

### Gestures chat (free part of the project)

In order to do curve matching, we have to do several steps. The first thing is to detect aruco codes in the video, which we do using the `cv2.aruco.ArucoDetector` class.

After detecting the codes, we have to get the points in the video that define the curve made by the movement of the code. We cannot simply get the center of the aruco code for every frame of the video, as this would generate too many points, so we define a minimum distance with the previous point for it to count as a distinct point. We also define a maximum distance to take care of outliers. Finally, we detect the start and the end of the sequence by detecting the first sequence of frames when the code is detected and the first sequence of frames when the code is not detected, so that if a person puts the code in their hand, they can start with the hand closed, begin the sequence by opening their hand, and end the sequence by closing the hand again.

Next, we have to detect which figure is being drawn. This was very tricky. At first, we decided to define each figure as a mathematical function (given by its parametrical representation; for example, a circle would be x^2 + y^2 - r^2) with parameters (so it could be fitted to the points), and using scipy we found which parameters better suited the points. That means, which parameters gave a lower MSE for the points in that function. However, it wasn't easy to find a good parametrization for every figure. Also, the biggest problem is that it was very common for it to find a local minimum (as the function for more complex shapes, such as the heart or the infinity, had a very high degree and plenty of inflexion points. Also, the global minimum could even not be the best real fit, as the function could have a very low point, close to 0 in the z axis, but far from a good fit, and the "good" fit could have a higher MSE).

To avoid this we decided against using mathematical functions all together. Our next idea was to just find the best fit by just finding the closest point in the other point cloud for each point and calculate the MSE with it. However, this didn't take into account the ordering of the points, and gave some very weird results.

Finally, we optimized that method by using Fréchet distance, which is a method to give a distance between two trajectories, i.e. it finds the distance between two ordered set of points. We still draw a base model for each figure, and then we use Fréchet distance to compare the model with the sequence of points. If the "distance" between them is lower than a certain threshold that we define, we consider the sequence of points to represent that figure. We also make it invariant to the starting point, the location of the points, and the scale.

![WandaVision](/images/report/heart_similar.png){:height="50%" width="50%"}![WandaVision](/images/report/heart_!similar.png){:height="50%" width="50%"}

Once we have the figure it represents, we play an audio linked with that figure that we have defined (for example, for the heart, it plays the sentence "I love you too"). We had to make threads for the audio player so that it runs in the background and the whole runtime isn't stopped when the audio is playing.

In the file `src/gestures_chat/pattern_creator.py` we create the model patterns and we adjust the threshold for each one, and in the file `src/gestures_chat/curve_matching.py` we run the described program in the Raspberri Pi.

## Results

The camera calibration works well. It detects all of the corners in the different positions and makes a good calibration. The pattern detection also works fine, as well as the sequence decoder. Lastly, the gestures chat works well, although if one closes their hand too many times with the aruco code in it, the paper may get crumpled, and the program may not recognize it well.

## Next steps

To add to the project, more figures could be added (with their respective audio files) so there are more than 4 and the chat is a bit more interactive. Some other ways to add to the project could be to add effects to the camera display when a pattern is detected (like some lightning bolts for the lightning).
