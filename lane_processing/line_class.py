import numpy as np

class line():
    def __init__(self):
        self.first_frame=False
        self.curvature=0

        self.right_fit=[np.array([False])]
        self.left_fit=[np.array([True])]
        self.max_tolerance=0.01

        self.img=None
        self.y_eval=700
        self.mid_x=640
        self.ym_per_pix=3.0/72.0
        self.xm_per_pix=3.7/650.0 #HardCoded

    def update_fit(self,left_fit,right_fit):
        if self.first_frame:
            error_left=((self.left_fit[0]-left_fit[0])**2).mean(axis=None)
            error_right=((self.right_fit[0]-right_fit[0])**2).mean(axis=None)
            if error_left<self.max_tolerance:
                self.left_fit=0.75*self.left_fit+0.25*left_fit
            if error_right<self.max_tolerance:
                self.right_fit=0.75*self.right_fit+0.25*right_fit

        else:
            self.right_fit=right_fit
            self.left_fit=left_fit

        self.update_curvature(self.right_fit)

    def update_curvature(self,fit):

        c1=(2*fit[0]*self.y_eval+fit[1])*self.xm_per_pix/self.ym_per_pix
        c2=2*fit[0]*self.xm_per_pix/(self.ym_per_pix**2)

        curvature=((1+c1*c1)**1.5)/(np.absolute(c2))

        if self.first_frame:
            self.curvature=curvature

        elif np.absolute(curvature-self.curvature)<500:
            self.curvature=0.75*self.curvature + 0.25* curvature

    def vehicle_position(self):
        left_pos=(self.left_fit[0]*(self.y_eval**2))+(self.left_fit[1]*self.y_eval)+self.left_fit[2]
        right_pos=(self.right_fit[0]*(self.y_eval**2))+(self.right_fit[1]*self.y_eval)+self.right_fit[2]

        return ((left_pos+right_pos)/2.0 - self.mid_x)*self.xm_per_pix