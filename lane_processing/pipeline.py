import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lane_processing.perspective import perspective_transform

# Step 1: Calculate the border mask
# Step 2: Perspective Transform
# Step 3: Lane extraction

REMOVE_CAR = False
CAR_BOX_TL = [150, 180]
CAR_BOX_BR = [256, 330]

# luminance formula (for finding track borders)
def compute_brightness(pixel):
    brightness = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    return brightness


def create_lane_mask(image, threshold=0.25, max_val=255):


    threshold = threshold * max_val
    
    # remove top rows
    # image[:105, :] = [max_val, max_val, max_val]  
    image[:105] = max_val  # grayscale

    if REMOVE_CAR:
        # remove the car from the mask
        # this is just a simple bounding box in which all pixels get set to the maximum value (always over the threshold)
        image[CAR_BOX_TL[0]:CAR_BOX_BR[0], CAR_BOX_TL[1]:CAR_BOX_BR[1], :] = [max_val, max_val, max_val]

    brightness_mask = (image < threshold).astype(np.uint8) * max_val

    return brightness_mask.astype(np.float32)

def lane_extraction(binary_warped):

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 100

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]

        good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right lane pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Check if either lane array is empty
    if (len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0):
        # we found no border, either on the left or right side!
        # just return None and deal with it later
        return None
    else:
        # Fit second-order polynomials
        left_fit  = np.polyfit(lefty,  leftx,  2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Compute lane curvature or offset
        y_eval = 700
        mid_x  = 640
        ym_per_pix = 3.0 / 72.0
        xm_per_pix = 3.7 / 650.0

        # Example offset from center
        left_pos  = left_fit[0]* (y_eval**2) + left_fit[1]* y_eval + left_fit[2]
        right_pos = right_fit[0]*(y_eval**2) + right_fit[1]* y_eval + right_fit[2]
        dx_num    = ((left_pos + right_pos)/2 - mid_x) * xm_per_pix

    # if dx_num > 0: turn right
    # if dx_num < 0: turn left

    return dx_num

def lane_extraction_image(binary_warped, original_image):

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 100

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]

        good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right lane pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Check if either lane array is empty
    if (len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0):
        # we found no border, either on the left or right side!
        # just return None and deal with it later
        return None, None
    else:
        # Fit second-order polynomials
        left_fit  = np.polyfit(lefty,  leftx,  2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx  = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0]* ploty**2 + right_fit[1]* ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left  = np.array([np.transpose(np.vstack([left_fitx,  ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

        original_image = (original_image * 255).astype('uint8')

        # just temporarily add the images like this
        # i am getting errors with addWeighted
        result = original_image
        # result = cv2.addWeighted(original_image, 1, color_warp, 0.3, 0)

        # Compute lane curvature or offset
        y_eval = 700
        mid_x  = 640
        ym_per_pix = 3.0 / 72.0
        xm_per_pix = 3.7 / 650.0

        # Example offset from center
        left_pos  = left_fit[0]* (y_eval**2) + left_fit[1]* y_eval + left_fit[2]
        right_pos = right_fit[0]*(y_eval**2) + right_fit[1]* y_eval + right_fit[2]
        dx_num    = ((left_pos + right_pos)/2 - mid_x) * xm_per_pix

        # Decide text: left or right of center
        text = "Right" if dx_num > 0 else "Left"
        dx_text = f"{text} ({dx_num:.2f})"

    # Show dx as text on the result
    cv2.putText(
        result,
        dx_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2
    )
    # if dx_num > 0: turn right
    # if dx_num < 0: turn left

    return dx_num, out_img
    
def pipeline(image):

    # 1. generate the lane mask
    mask_lane = create_lane_mask(image)

    # 2. transform the perspective
    mask_transformed, _, _ = perspective_transform(mask_lane, None, None)

    # 3. generate the lane prediction from the frame and lane_mask
    dx = lane_extraction(mask_transformed)
    
    return dx

if __name__ == "__main__":
    image_path = './test_images_color/output2.png'
    image = mpimg.imread(image_path)

    mask_lane = create_lane_mask(image)

    mask_transformed, _, _ = perspective_transform(mask_lane, None, None)

    fig, axes=plt.subplots(1,3, figsize=(20,10))
    axes=axes.ravel()

    test_img = image.copy()
    axes[0].imshow(test_img, cmap='gray')
    test_img = np.dstack((test_img, test_img, test_img))