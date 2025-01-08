import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lane_processing.colour_and_gradient import gradient_color_thresh


def transformation_one(height:int = 256, width:int = 512):
    center_width = width/2

    center_top_diff = 100
    center_bottom_diff = width/2 -10

    height_before_car = height * 0.60
    height_before_car_offset = -1 * 50

    dst_center_top_diff = 100
    dst_center_bottom_diff = 100
    dst_top = 0
    dst_bottom = height

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
    return src, dst

def transformation_two(height: int = 256):

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
    
    return src, dst


def perspective_transform(
        image,
        src = None,
        dst = None
        ):

    if src is None and dst is None:
        src, dst = transformation_one()

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size=(image.shape[1],image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped,Minv,M



if __name__ == "__main__":
    image_path = './test_images_color/output2.png'
    image = mpimg.imread(image_path)
    image = gradient_color_thresh(image)

    src, dst = transformation_one()
    binary_warped,Minv,M = perspective_transform(image, src=src, dst=dst)#

    fig, axes=plt.subplots(1,3, figsize=(20,10))
    axes=axes.ravel()

    test_img = image.copy()
    axes[0].imshow(test_img, cmap='gray')
    test_img = np.dstack((test_img, test_img, test_img))

    for point in src.astype(int):
        print(tuple(point.tolist()))
        # ...
        cv2.circle(test_img,tuple(point.tolist()),10,(255,0,0),-1) # rot
    print()
    for point in dst.astype(int):
        print(tuple(point.tolist()))
        # ...
        cv2.circle(test_img,tuple(point.tolist()),10,(0,255,255),-1) # grün


    axes[1].imshow(test_img)
    axes[2].imshow(binary_warped, cmap='gray')
    plt.axis('off')
    plt.show()