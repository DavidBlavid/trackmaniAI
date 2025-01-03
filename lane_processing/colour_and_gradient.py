import numpy as np

# Define a function to compute brightness using luminance formula
def compute_brightness(pixel):
    # Convert RGB to brightness using standard luminance weights
    brightness = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    return brightness

image_path = 'test_images_color/output9.png'
image = mpimg.imread(image_path)

processed = image.copy()

# set any pixel below x=105 to white
processed[:105, :] = [1, 1, 1]

# remove the car, as it has the same color as the border
# we just fill the square in which it is to white, same as above
car_tl = [150, 180]
car_br = [256, 330]
processed[car_tl[0]:car_br[0], car_tl[1]:car_br[1], :] = [1, 1, 1]

# Apply brightness calculation and thresholding
threshold = 0.25  # adjust threshold here
height, width, _ = processed.shape
brightness_mask = np.zeros((height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        brightness = compute_brightness(processed[y, x, :])
        brightness_mask[y, x] = 255 if brightness < threshold else 0

fig, (ax1, ax2)=plt.subplots(1,2, figsize=(10,10))
ax1.set_title('Original Image')
ax1.imshow(image)
ax2.set_title('Brightness Mask')
ax2.imshow(brightness_mask, cmap='gray')
plt.show()

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image_path = 'test_images_color/output6.png'
image = mpimg.imread(image_path)

processed = image.copy()
processed[:105, :] = [1, 1, 1]

# Define a function to compute brightness using luminance formula
def compute_brightness(pixel):
    brightness = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    return brightness

# Apply brightness calculation and thresholding
threshold = 0.25  # Adjust this value to debug and find the perfect threshold
height, width, _ = processed.shape
brightness_mask = np.zeros((height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        brightness = compute_brightness(processed[y, x, :])
        brightness_mask[y, x] = 255 if brightness < threshold else 0




def abs_sobel_thresh(image,orient,  thresh=(20, 100)):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    if orient=='x':x, y=1,0
    else: x, y=0,1

    sobel=cv2.Sobel(gray,cv2.CV_64F,x,y)
    sobel=np.absolute(sobel)
    scaled_sobel=np.uint8(255*sobel/np.max(sobel))

    sx_binary=np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])]=1
    binary_output=np.copy(sx_binary)
    return binary_output


def mag_thresh(img, mag_thresh=(20,150)):

    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0)
    sobely=cv2.Sobel(gray,cv2.CV_64F,0,1)
    sobel=np.sqrt(np.square(sobelx)+np.square(sobely))
    scaled_sobel=np.uint8(255*sobel/np.max(sobel))

    # t=sum((i > 150) &(i<200)  for i in scaled_sobel)
    binary_sobel=np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel>=mag_thresh[0]) & (scaled_sobel<=mag_thresh[1])]=1
    return binary_sobel

def dir_threshold(img,  thresh=(0.7,1.3)):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx=np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    sobely=np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1))

    dir_=np.arctan2(sobely,sobelx)

    sx_binary = np.zeros_like(gray)
    sx_binary[(dir_>=thresh[0]) &(dir_<=thresh[1])]=1
    binary_output=sx_binary
    return binary_output

# ## Thresholding on HLS color channels.
#
# 1. Thesholding on S and L channel for detecting Yellow Lane lines.
#
# 2. Thresholding on L channel to detect white lane lines.
def color_space(image,thresh=(170,255)):
    hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    l_channel=hls[:,:,1]
    s_channel=hls[:,:,2]
    s_binary=np.zeros_like(s_channel)

    _, gray_binary = cv2.threshold(gray_image.astype('uint8'), 100, 255, cv2.THRESH_BINARY)
    s_binary[(s_channel>=thresh[0]) & (s_channel<=thresh[1])&(l_channel>=80)]=1
    color_output=np.copy(s_binary)
    return color_output

def segregate_white_line(image,thresh=(200,255)):
    hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    l_channel=hls[:,:,1]
    l_binary=np.zeros_like(l_channel)
    l_binary[((l_channel>=200)&(l_channel<=255))]=1
    return l_binary


# ## Combining gradient and color thresholding

def gradient_color_thresh(image):
    ksize=3
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 200))
    grady = abs_sobel_thresh(image, orient='y', thresh=(20, 200))

    mag_binary = mag_thresh(image, mag_thresh=(20, 200))

    dir_binary = dir_threshold(image, thresh=(0.7, 1.3))

    color_binary=color_space(image,thresh=(100,255))

    combined = np.zeros_like(dir_binary)
    combined[(color_binary==1)|((gradx == 1)& (grady == 1)) |(mag_binary==1) &(dir_binary==1)] = 1

    kernel = np.ones((45,45),np.uint8)
    start_morph = 0
    morph_image=combined[start_morph:,:950]
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
    combined[start_morph:,:950]=morph_image

    white_line=segregate_white_line(image,thresh=(200,255))
    combined= cv2.bitwise_or(combined,white_line)
    return combined