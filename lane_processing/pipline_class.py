import cv2
import numpy as np

from lane_processing.calibrate_cam import undistort
from lane_processing.colour_and_gradient import gradient_color_thresh


class pipeline:
    line=None
    M=None
    Minv=None
    camera_mtx=None
    dist=None

    @staticmethod
    def set_vals(l,camera_mtx,dist,m,minv):
        pipeline.line=l
        pipeline.camera_mtx=camera_mtx
        pipeline.dist=dist
        pipeline.M=m
        pipeline.Minv=minv

    @staticmethod
    def Pipeline(get_frame,time):
        img = get_frame(time)
        # print(img)
        line,M,Minv,camera_mtx,dist=pipeline.line,pipeline.M,pipeline.Minv,pipeline.camera_mtx,pipeline.dist

        img_size=(img.shape[1],img.shape[0])
        width,height=img_size

        undist_img=undistort(np.copy(img),camera_mtx,dist)
        thresh_image=gradient_color_thresh(undist_img)

        binary_warped = cv2.warpPerspective(thresh_image, M, img_size, flags=cv2.INTER_LINEAR)

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        nwindows = 9

        window_height = int(binary_warped.shape[0]/nwindows)

        margin = 100
        minpix=50

        if not line.first_frame:
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
            # Create an output image to draw on and  visualize the result
            midpoint = int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
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

            line.update_fit(left_fit,right_fit)
            line.first_frame=True

        else:
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_fit=line.left_fit
            right_fit=line.right_fit

            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
            left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
            left_fit[1]*nonzeroy + left_fit[2] + margin)))

            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
            right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
            right_fit[1]*nonzeroy + right_fit[2] + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            line.update_fit(left_fit,right_fit)
            left_fit=line.left_fit
            right_fit=line.right_fit

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        dx=line.vehicle_position()
        if dx>0:
            text='Left'
        else:
            text='Right'

        # cv2.putText(result,'Radius of curvature  = %.2f m'%(line.curvature),(20,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        #
        # cv2.putText(result,'Vehicle position : %.2f m %s of center'%(abs(dx), text),(20,90),
        #                 font, 1,(255,255,255),2,cv2.LINE_AA)
        return result