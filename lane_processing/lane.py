
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
from lane_processing.calibrate_cam import undistort
from lane_processing.colour_and_gradient import gradient_color_thresh

# Visualizing on a test Image

thresh_image=gradient_color_thresh(undist*255)
print(thresh_image.max())
fig, axes=plt.subplots(1,2, figsize=(15,10))
axes[0].imshow(undist, cmap='gray')
axes[1].imshow(thresh_image, cmap='gray')
# plt.axis('off')
# cv2.imwrite('output_images/gradient_color_thresh.jpg', thresh_image)



# ## Perspective Transform

def perspective_transform(
        image,
        src=np.float32([[195,720],[590,460],[700,460],[1120,720]]),
        dst=np.float32([[350,720],[410,0],[970,0],[1000,720]])
        ):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size=(image.shape[1],image.shape[0])
    # print(M.shape,Minv.shape)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped,Minv,M

# ### Visualiziing Perspective Transform

height, width = thresh_image.shape
center_width = width/2

center_top_diff = 300
center_bottom_diff = 500

height_before_car = height * 0.60
height_before_car_offset = -1 * 100

dst_center_top_diff = 300
dst_center_bottom_diff = 300
dst_top = 0
dst_bottom = height

ansatz = 1

if ansatz == 1:
    src=np.float32([
        [center_width-center_bottom_diff,height_before_car], # unten links
        [center_width-center_top_diff,height_before_car + height_before_car_offset], # oben links
        [center_width+center_top_diff,height_before_car + height_before_car_offset], # oben rechts
        [center_width+center_bottom_diff,height_before_car] # unten rechts
    ])
    dst=np.float32([
        [center_width - dst_center_bottom_diff,dst_bottom], # unten links
        [center_width - dst_center_top_diff, dst_top], # oben links
        [center_width + dst_center_top_diff, dst_top], # oben rechts
        [center_width + dst_center_bottom_diff,dst_bottom] # unten rechts
    ])
elif ansatz == 2:
    line_dst_offset = 200
    src = np.float32([
        [595, 452],
        [685, 452],
        [1110, height],
        [220, height]])

    dst = np.float32([
        [src[3][0] + line_dst_offset, 0],
        [src[2][0] - line_dst_offset, 0],
        [src[2][0] - line_dst_offset, src[2][1]],
        [src[3][0] + line_dst_offset, src[3][1]]])

print(src)
print(dst)

binary_warped,Minv,M=perspective_transform(thresh_image, src=src, dst=dst)#

fig, axes=plt.subplots(1,3, figsize=(20,20))
axes=axes.ravel()

test_img = thresh_image.copy()
axes[0].imshow(test_img, cmap='gray')
test_img = np.dstack((test_img, test_img, test_img))

for point in src.astype(int):
    # print(tuple(point.tolist()))
    # ...
    cv2.circle(test_img,tuple(point.tolist()),20,(255,0,0)) # rot

for point in dst.astype(int):
    # print(tuple(point.tolist()))
    # ...
    cv2.circle(test_img,tuple(point.tolist()),20,(0,255,255)) # grün


axes[1].imshow(test_img)
axes[2].imshow(binary_warped, cmap='gray')
plt.axis('off')

# ## Pipeline to test Image processing and Output on the final Image.

def pipeline(binary_warped,count,image,undist_image):
    debug_grey = binary_warped.copy()
    tmp_test = np.zeros_like(image)
    tmp_test[:,:,0] = debug_grey * 255
    tmp_test[:,:,1] = debug_grey * 255
    tmp_test[:,:,2] = debug_grey * 255
    print(binary_warped.shape)

    if count==0:
        # After creating a warped binary Image,
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0) # original
        # histogram = np.sum(binary_warped, axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9

        # Set height of windows
        window_height = int(binary_warped.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 100

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            # win_y_low = (window+1)*window_height
            # win_y_high = window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            print(good_left_inds.shape , good_right_inds.shape)
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # print(color_warp[0].shape)
        # print(color_warp.shape)
        # tmp_test += color_warp.copy()

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)

        y_eval=700
        mid_x=640
        ym_per_pix=3.0/72.0
        xm_per_pix=3.7/650.0

        c1=(2*right_fit[0]*y_eval+right_fit[1])*xm_per_pix/ym_per_pix
        c2=2*right_fit[0]*xm_per_pix/(ym_per_pix**2)

        curvature=((1+c1*c1)**1.5)/(np.absolute(c2))

        left_pos=(left_fit[0]*(y_eval**2))+(left_fit[1]*y_eval)+left_fit[2]
        right_pos=(right_fit[0]*(y_eval**2))+(right_fit[1]*y_eval)+right_fit[2]

        dx=((left_pos+right_pos)/2-mid_x)*xm_per_pix
        if dx>0:
            text='Left'
        else:
            text='Right'

        font=cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(result,'Radius of curvature  = %.2f m'%(curvature),(20,50), font, 1,(255,255,255),2,cv2.LINE_AA)
#
        # cv2.putText(result,'Vehicle position : %.2f m %s of center'%(abs(dx), text),(20,90),
        #                 font, 1,(255,255,255),2,cv2.LINE_AA)

        return result, tmp_test , (left_fitx, right_fitx ), ploty, histogram, out_img # tmp_test ist von mir zum debuggen

count=0
result, test, curves, ploty, hist, out_img = pipeline(binary_warped,count,image,undist)
# print(curves)
fig, axes = plt.subplots(2,2, figsize=(10,10))
axes = axes.flatten()
axes[0].plot(hist)
axes[1].imshow(result)
axes[2].imshow(out_img)
axes[2].plot(curves[0], ploty, color='yellow', linewidth=3)
axes[2].plot(curves[1], ploty, color='yellow', linewidth=3)
plt.axis('off')
plt.show()

def frame_transformation(img, max_val = 255):
    """
    Applies the transformation to a single frame.
    1. Sets pixels below y=105 to white.
    2. Removes the car by filling a bounding box with white.
    3. Applies brightness calculation and thresholding.
    """
    processed = img.copy()

    # Set any pixel below x=105 to white
    processed[:105, :] = [max_val, max_val, max_val]

    # Remove the car by filling its bounding box with white
    car_tl = [150, 200]
    car_br = [230, 300]
    processed[car_tl[0]:car_br[0], car_tl[1]:car_br[1], :] = [max_val, max_val, max_val]

    # Apply brightness calculation and thresholding
    # threshold = 0.5  # Adjust threshold here
    threshold = 0.25 * 255
    height, width, _ = processed.shape
    brightness_mask = np.zeros((height, width), dtype=np.uint8)

    # Compute brightness and threshold
    for y in range(height):
        for x in range(width):
            brightness = 0.299 * processed[y, x, 0] + 0.587 * processed[y, x, 1] + 0.114 * processed[y, x, 2]
            brightness_mask[y, x] = max_val if brightness < threshold else 0

    # Return the binary mask for debugging or overlaying
    return np.stack([brightness_mask] * 3, axis=-1)  # Stack to match the 3-channel format

def pipeline(get_frame, time):
    img = get_frame(time)
    return frame_transformation(img)

from moviepy import VideoFileClip

video_output = 'output_videos/video2.mp4'
clip1 = VideoFileClip("./video/video.mp4")

# Apply the transformation frame by frame
video_clip = clip1.transform(func = pipeline)

# Write the transformed video
video_clip.write_videofile(video_output, audio=False)



video_output = 'output_videos/video2.mp4'
clip1 = VideoFileClip("./video/video_scaled.mp4")
video_clip = clip1.transform(pipeline.Pipeline, apply_to='mask')

first_frame = clip1.get_frame(0)

# Apply transformation
transformed_frame = frame_transformation(first_frame, max_val=255)

# Debug: Show original and transformed frame
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_title("Original Frame")
ax1.imshow(first_frame)
ax2.set_title("Transformed Frame")
ax2.imshow(transformed_frame, cmap='gray')
plt.show()

# debug video files with this


try:
    info = ffmpeg_parse_infos("./video/video.mp4")
    print(info)
except Exception as e:
    print(f"Error: {e}")


# luminance formula
def compute_brightness(pixel):
    brightness = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    return brightness

def create_lane_mask(frame, threshold=0.25, max_val=255):
    processed = frame.copy()
    threshold = threshold * max_val
    processed[:105, :] = [max_val, max_val, max_val]  # remove top rows

    # remove the car from the mask
    # this is just a simple bounding box in which all pixels get set to the maximum value (always over the threshold)
    car_tl = [150, 180]
    car_br = [256, 330]
    processed[car_tl[0]:car_br[0], car_tl[1]:car_br[1], :] = [max_val, max_val, max_val]

    height, width, _ = processed.shape
    brightness_mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            brightness = compute_brightness(processed[y, x, :])
            brightness_mask[y, x] = max_val if brightness < threshold else 0

    # binary_mask = (brightness_mask == max_val).astype(np.float32)
    binary_mask = brightness_mask.astype(np.float32)

    return binary_mask

# ----------------------------------------------------------------------
# 2) The pipeline that uses the mask (binary_warped) to detect lanes.
# ----------------------------------------------------------------------
def pipeline(binary_warped, count, image, undist_image):
    debug_grey = binary_warped.copy()
    tmp_test = np.zeros_like(image)
    tmp_test[:,:,0] = debug_grey * 255
    tmp_test[:,:,1] = debug_grey * 255
    tmp_test[:,:,2] = debug_grey * 255

    if count == 0:
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
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0,255,0), 2)

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
            # If so, skip polynomial fitting and lane drawing
            dx_text = "N.A."
            result = undist_image.copy()  # leave frame unchanged
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
            result = cv2.addWeighted(undist_image, 1, color_warp, 0.3, 0)

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

        return result, tmp_test, (None, None), None, histogram, out_img

def video_mask(get_frame, time):

    frame = get_frame(time)

    # generate the brightness mask
    lane_mask = create_lane_mask(frame)

    return lane_mask

    # undist_image = frame.copy()

    # generate the lane prediction from the frame and lane_mask
    # result, debug_test, (lfx, rfx), py, hist, out_img = pipeline(lane_mask, 0, frame, undist_image)
    # return result

# process a video
if __name__ == "__main__":
    video_input_path = "video/20s.mp4"
    video_output_path = "output_videos/mask1.mp4"

    original_clip = VideoFileClip(video_input_path).subclipped(0,6)
    video_clip = original_clip.transform(func=video_mask)
    video_clip.write_videofile(video_output_path, audio=False)


from moviepy import VideoFileClip

video_output = 'output_videos/video2.mp4'
clip1 = VideoFileClip("./video/video.mp4")

# Apply the transformation frame by frame
video_clip = clip1.transform(func = pipeline)

# Write the transformed video
video_clip.write_videofile(video_output, audio=False)
