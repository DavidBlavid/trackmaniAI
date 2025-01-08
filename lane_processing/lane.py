
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos

from lane_processing.calibrate_cam import undistort
from lane_processing.colour_and_gradient import gradient_color_thresh
from lane_processing.pipeline import pipeline

## Perspective Transform

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




video_output = 'output_videos/video2.mp4'
clip1 = VideoFileClip("./video/video.mp4")

# Apply the transformation frame by frame
video_clip = clip1.transform(func = pipeline)

# Write the transformed video
video_clip.write_videofile(video_output, audio=False)
