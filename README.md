# **Project: Finding Lane Lines on the Road** 
---
## **Goals / Steps**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Detect lane lines in any color
* Detect continuous straight lines or dash lines
* Keep algorithm robust enough to identify lane lines on complicated road background, such as shadows, dirty road surface.

The output picture with marked lane lines look like this:
![](https://github.com/uranus4ever/Lane-Detection/master/LaneDectionExamplePic.png)  


## Usage
If you do not install python packages, the following files could still be opened as HTML to review the coding ideas and outputs:
* P1. ipynb - Normal mode with clear road lanes
* P1_challenge.ipynb - Challenge mode with unclear road lanes correspond to challenge.mp4 video 

The source code files are as below, which require Python 3.5 and related packages:
* Lane-project.py - Normal mode
* Lane-project-challenge.py - Challenge mode

---

### Reflection

### 1. Pipeline. 

My pipeline consisted of 7 steps. First, I converted the images to grayscale, then I blur the images with Gaussian filter. After that, edges could be detected by Canny function in OpenCV package. For this project, we only care about the lines on the road, as a result, I plot a quadrilateral mask to show Region of Interest. Within that mask, a Hough tranfsformation is applied to generate lines. Then I draw red lines to mark lanes. Finally I merge the origin input image and marked lines image into the output image.

The following picture shows the main process of pipeline: 

![](https://github.com/uranus4ever/Lane-Detection/master/PipelineProcess.jpg)

To improve the detection result, I tune the Canny function with multiple parameters set. It is visualized as th following picture:

![](https://github.com/uranus4ever/Lane-Detection/master/CannyParameters.png)

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
![](https://github.com/uranus4ever/Lane-Detection/master/Comparison_colorfilter.png)

### 2. Potential shortcomings with current pipeline

One potential shortcoming would be what would happen when the lane lines are not clear due to worn or on dirty background. Or, in other cases, if there are more than one line on one side on the road, the pipeline will be confused. For example, the lanes are modified to another position but old ones can still be identified. 

### 3. Suggest possible improvements

A possible improvement would be to merge the parameters matrix to fit all scenarios.

* Scenario 1 - normal mode
```
kernel_size = 5
low_threshold = 50
high_threshold = 150
min_line_len = 20
max_line_gap = 100
```

* Scenario 2 - challenge mode
```
kernel_size = 7
low_threshold = 150
high_threshold = 250
min_line_len = 5
max_line_gap = 100
```
