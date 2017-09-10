'''
challenge video
'''

# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# matplotlib inline
import math
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# from IPython.display import HTML

def YellowFilter(img):
    Yellow2White = np.copy(img)
    red_threshold = 200
    green_threshold = 50
    blue_threshold = 0
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    threshold = (img[:,:,0] > rgb_threshold[0]) \
                 & (img[:,:,1] > rgb_threshold[1]) \
                 & (img[:,:,2] > rgb_threshold[2])
    Yellow2White[threshold] = [250, 250, 250]
    return Yellow2White

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color, thickness):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if len(lines) < 1:
        return img
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_one_line(img, lines, color, thickness):
    if len(lines) > 1:
        lines_array = np.vstack(lines)
        x1 = np.min(lines_array[:, 0]) # index 0,1,2,3 correspond to 2 points (x1, y1), (x2, y2)
        x2 = np.max(lines_array[:, 2])
        y1 = lines_array[np.argmin(lines_array[:, 0]), 1]
        y2 = lines_array[np.argmax(lines_array[:, 2]), 3]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                            np.array([]), min_line_len, max_line_gap)

    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            if fit[0] < -0.5:
                left_lines.append(line)
            elif fit[0] > 0.5:
                right_lines.append(line)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_one_line(line_img, left_lines, color=(255,0,0), thickness=10)
    draw_one_line(line_img, right_lines, color=(255,0,0), thickness=10)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α, β, λ):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    YFPic = YellowFilter(image)
    gray = grayscale(YFPic)
    blur_gray = gaussian_blur(gray, kernel_size=7)
    edges = canny(blur_gray, low_threshold=150, high_threshold=250)
    imshape = image.shape
    top_left, top_right = (imshape[1] * 0.48, imshape[0] * 0.6), (imshape[1] * 0.52, imshape[0] * 0.6)
    vertices = np.array([[(0, imshape[0]), top_left, top_right, (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40  # minimum number of pixels making up a line
    max_line_gap = 200  # maximum gap in pixels between connectable line segments

    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    result = weighted_img(line_img, image, α=0.8, β=1., λ=0.)
    return result

pic = mpimg.imread('challenge_pic_clean.jpg')
output = process_image(pic)
plt.imshow(output)
plt.show()

white_output = 'C:/Users/scq/carnd-lanelines-p1/test_videos_output/challenge_advanced.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
input_path = 'C:/Users/scq/carnd-lanelines-p1/test_videos/challenge.mp4'
# clip1 = VideoFileClip(input_path).subclip(3,7)
clip1 = VideoFileClip(input_path)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

# #########
# Debug
# #########

# plt.figure()
# plt.subplot(221)
# plt.title('3, 100')
# plt.imshow(process_image(pic, 3, 100))
# plt.subplot(222)
# plt.title('40, 60')
# plt.imshow(process_image(pic, 40, 60))
# plt.subplot(223)
# plt.title('3, 200')
# plt.imshow(process_image(pic, 3, 200))
# plt.subplot(224)
# plt.title('40, 200')
# plt.imshow(process_image(pic, 40, 200))
# plt.show()

'''
# YFpic = YellowFilter(pic)
# gray = grayscale(YFpic)
# blur_gray = gaussian_blur(gray, kernel_size=7)
# edges = canny(blur_gray, low_threshold=100, high_threshold=200)
# imshape = pic.shape
# top_left, top_right = (imshape[1] * 0.48, imshape[0] * 0.6), (imshape[1] * 0.52, imshape[0] * 0.6)
# vertices = np.array([[(0, imshape[0]), top_left, top_right, (imshape[1], imshape[0])]], dtype=np.int32)
# masked_edges = region_of_interest(edges, vertices)
#
# edges_1 = canny(blur_gray, low_threshold=150, high_threshold=250)
# masked_edges_1 = region_of_interest(edges_1, vertices)
#
# edges_2 = canny(blur_gray, low_threshold=50, high_threshold=150)
# masked_edges_2 = region_of_interest(edges_2, vertices)
#
# plt.figure()
# plt.subplot(221)
# plt.imshow(blur_gray, cmap='gray')
# plt.title('Blur_With Color Filter')
# plt.subplot(222)
# plt.imshow(masked_edges)
# plt.title('100, 200')
# plt.subplot(223)
# plt.imshow(masked_edges_1)
# plt.title('150, 250')
# plt.subplot(224)
# plt.imshow(masked_edges_2)
# plt.title('50, 150')
# plt.show()
'''



