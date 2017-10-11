# **Project: Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: ./Img/PipelineProcess.jpg "PipelineProcess"
[image2]: ./Img/CannyParameters.png "Canny Parameters"
[image3]: ./Img/Comparison_colorfilter.png "Color Filter"
[video1]: ./Img/solidWhite.mp4 "Video1"
[video2]: ./Img/solidYellowLeft.mp4 "Video2"
[video3]: ./Img/challenge.mp4 "Video3-challenge"
[gif1]: ./Img/challenge.gif "Challenge_gif"


## **Goals / Steps**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Detect lane lines in any color
* Detect continuous straight lines or dash lines
* Keep algorithm robust enough to identify lane lines on complicated road background, such as shadows, dirty road surface.

The output video with marked lane lines look like this gif:
![alt text][gif1]


## Usage
If you do not install python packages, the following Jupyter Notebook file could still be opened as HTML to review the coding ideas and output videos. 
The source code Python file is uploaded as well, which requires Python 3.5 and related packages.

| Code          | Output Video | Comments |
| :---:         |     :---:      |         :---: |
| P1.ipynb <br>Lane-project-challenge.py   | solidWhiteRight.mp4 <br>solidYellowLeft.mp4 <br>challenge.mp4     | The source code fits all three output videos   |

## Dependencies
* Numpy
* cv2
* moviepy

---

### Reflection

### 1. Pipeline. 

My pipeline consisted of 7 steps. First, I converted the images to grayscale, then I blur the images with Gaussian filter. After that, edges could be detected by Canny function in OpenCV package. For this project, we only care about the lines on the road, as a result, I plot a quadrilateral mask to show Region of Interest. Within that mask, a Hough tranfsformation is applied to generate lines. Then I draw red lines to mark lanes. Finally I merge the origin input image and marked lines image into the output image.

The following picture shows the main process of pipeline: 

![alt text][image1]

To improve the detection result, I tune the Canny function with multiple parameters set. It is visualized as th following picture:

![alt text][image2]

And I calibrate the key parameters to fit all scenarios in the video.

```
kernel_size = 7
low_threshold = 150
high_threshold = 250
min_line_len = 5
max_line_gap = 100
```

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by SlopeFilter and LineMerge:
* SlopeFilter:
```python
for line in lines:
    for x1, y1, x2, y2 in line:
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        if fit[0] < -0.5:
            left_lines.append(line)
        elif fit[0] > 0.5:
            right_lines.append(line)
```

* LineMerge:
```
if len(lines) > 1:
    lines_array = np.vstack(lines)
    x1 = np.min(lines_array[:, 0]) # index 0,1,2,3 correspond to 2 points (x1, y1), (x2, y2)
    x2 = np.max(lines_array[:, 2])
    y1 = lines_array[np.argmin(lines_array[:, 0]), 1]
    y2 = lines_array[np.argmax(lines_array[:, 2]), 3]
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
```
        
In the challenge video, one of the obstale to identify yellow lane line due to tree shadow, dirty road surface and worn lines. I use a ColorSelection to turn yellow pixel into white before sending image to grayscale.

```
red_threshold = 200
green_threshold = 50
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]
threshold = (img[:,:,0] > rgb_threshold[0]) \
             & (img[:,:,1] > rgb_threshold[1]) \
             & (img[:,:,2] > rgb_threshold[2])
Yellow2White[threshold] = [250, 250, 250]
```

The following comparison picture clearly shows the practical effectiveness of ColorSelection:

![alt text][image3]

### 2. Output Videos

Does the pipeline established with the test images work to process the video?
<br>It sure does! Here's a [link to my video](./solidWhiteRight.mp4). It looks stable, isn't it?

The above video is too easy? Let's try something harder - [yellow lane](./solidYellowLeft.mp4).

The real challenge is coming! The next video combines daylight and shadow on the road, which requires algorithm robust enought to cover complex conditions. [See Challenge video](./challenge.mp4).

### 3. Potential shortcomings with current pipeline

One potential shortcoming would be what would happen when the lane lines are not clear due to worn or on dirty background. Or, in other cases, if there are more than one line on one side on the road, the pipeline will be confused. For example, the lanes are modified to another position but old ones can still be identified. 

### 4. Suggest possible improvements

A possible improvement would be to create the link between two continuous picture in the video. That would help to make output detection more natural if eliminating sudden appearance/disappearance.

There are other solutions here. Please consider the following,
threshold = 50
min_line_len = 100
max_line_gap = 160

Please examine the parameters by modifying them separately. That will allow you to identify how each parameter affects the lane line. For example,

'max_line_gap' defined the maximum distance between segments that will be connected to a single line.
'min_line_len' defined the minimum length of a line that will be created.
Increasing these parameters will create smoother and longer lines

"threshold" defined the minimum number of intersections in a given grid cell that are required to choose a line.
Increasing this parameter, the filter will choose longer lines and ignore short lines.
