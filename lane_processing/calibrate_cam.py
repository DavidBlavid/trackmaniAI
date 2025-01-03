import glob
import pickle

import matplotlib.pyplot as plt
import cv2
from re import split
import numpy as np


def calibrate_camera(image_path_list:list[str] = None):
    if image_path_list is not None:
        tmp = cv2.imread(image_path_list[0]).shape
        img_size = (tmp[1], tmp[0])
    else:
        raise ValueError("No image path list provided")

    objpoints = []
    imgpoints = []  # 2d points in image plane.

    fig, axs = plt.subplots(3, 5, figsize=(16, 11))
    axs = axs.ravel()

    # Find Chessboard corners for every image.
    for i, fname in enumerate(image_path_list):
        img = cv2.imread(fname)
        # axs[0].imshow(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # axs[1].imshow(gray)

        # get chessboard info from file name
        # pp -> points per
        string_array_data = split(r'[_.]', fname)[-3:-1]
        pp_row, pp_col = tuple(map(int, string_array_data))
        # Find the chessboard corners
        # print(f"image {fname} mit dem Tupel {int_tuple}")

        ret, corners = cv2.findChessboardCorners(gray, (pp_row, pp_col), None)

        # If found, add object points, image points
        if ret == True:
            # Object Points- represents the 3D object in the real world space.
            objp = np.zeros((pp_row * pp_col, 3), np.float32)
            objp[:, :2] = np.mgrid[0:pp_row, 0:pp_col].T.reshape(-1, 2)

            objpoints.append(objp)

            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (pp_row, pp_col), corners, ret)
            axs[i].axis('off')
            axs[i].imshow(img)
            cv2.imwrite('output_images/corners_drawn' + str(i) + '.jpg', img)
        else:
            axs[i].set_title(fname)
            axs[i].imshow(img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return ret, mtx, dist

def undistort(img, mtx = None, dist = None):
    if None not in(mtx, dist):
        try:
            with open("./calibration.p", "rb") as f:
                pickle_object = pickle.load(f)
                mtx = pickle_object["mtx"]
                dist = pickle_object["dist"]
        except FileNotFoundError:
            print("No calibration file found. Please run calibrate_camera() first.")

    undist=cv2.undistort(img, mtx, dist, None, mtx)
    return undist


if __name__ == '__main__':
    images = glob.glob('./camera_calc/*.jpeg')
    ret, mtx, dist = calibrate_camera(image_path_list=images)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./calibration.p", "wb"))

