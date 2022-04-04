#Base code by Pratik Chaudhari (pratikac@seas.upenn.edu)
#Implementation by Roshan Benefo

import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *
import cv2
import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(self, resolution=0.05):
        self.resolution = resolution
        self.xmin, self.xmax = -20, 20
        self.ymin, self.ymax = -20, 20
        self.szx = int(np.ceil((self.xmax-self.xmin)/self.resolution+1))
        self.szy = int(np.ceil((self.ymax-self.ymin)/self.resolution+1))
        
        # binarized map and log-odds
        self.cells = np.zeros((self.szx, self.szy), dtype=np.int8)
        self.log_odds = np.zeros(self.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        self.log_odds_max = 5e6
        # number of observations received yet for each cell
        self.num_obs_per_cell = np.zeros(self.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        self.occupied_prob_thresh = 0.6
        self.log_odds_thresh = np.log(self.occupied_prob_thresh/(1-self.occupied_prob_thresh))

    def grid_cell_from_xy(self, x, y, indiv = False):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situationself.
        """
        if not indiv:
            ret = np.zeros((2,len(x)))
        else:
            ret = np.zeros((2,1))
        #first, shift (0,0), which is centered in midplane, to edge of map. Let's do upper left corner.

        ##HOW TO TAKE INTO ACCOUNT RESOLUTION??
        x_shifted = (x/self.resolution+self.szx/2).astype(int) #Add, because points that were (0, y) (in middle), should now be (positive #, y)
        y_shifted = (-y/self.resolution+self.szy/2).astype(int) #Add, but flip y (as y goes up currently, we go up in map, but we want to go down)
        x_clipped = np.clip(x_shifted, 0, self.szx-1)
        y_clipped = np.clip(y_shifted, 0, self.szy-1)
        ret[0] = x_clipped
        ret[1] = y_clipped
        return ret

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equationself.
    """
    def __init__(self, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        self.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        self.Q = Q

        # we resample particles if the effective number of particles
        # falls below self.resampling_threshold*num_particles
        self.resampling_threshold = resampling_threshold

        # initialize the map
        self.map = map_t(resolution)

    def read_data(self, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        self.idx = idx
        self.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        self.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        self.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(self.joint['t']-t))

    def init_sensor_model(self):
        # lidar height from the ground in meters
        self.head_height = 0.93 + 0.33
        self.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        self.lidar_dmin = 1e-3
        self.lidar_dmax = 30
        self.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        self.lidar_angles = np.arange(-135,135+self.lidar_angular_resolution,
                                   self.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cellself. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        self.lidar_log_odds_occ = np.log(9)
        self.lidar_log_odds_free = np.log(1/9.)

    def init_particles(self, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        self.n = n
        self.p = deepcopy(p) if p is not None else np.zeros((3,self.n), dtype=np.float64)
        self.w = deepcopy(w) if w is not None else np.ones(n)/float(self.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        i = 0
        new_p = np.zeros(p.shape)
        n = p.shape[1]
        r = np.random.uniform(0, 1/n)
        co = w[0]
        for m in range(n):
            u =  (r+m/n)
            while u > co:
                i += 1
                co = co+w[i]
            new_p[:,m] = p[:,i]

        w = np.ones(n)/n #equal weights
        return new_p, w

    def rays2world(self, p, d, head_pose, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply self.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, 
        for each ray (the length of d has to be equal to that of angles, this is self.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data

        useful_idxs = np.logical_and(d < self.lidar_dmax, d > self.lidar_dmin)
        good_lidar = d[useful_idxs]
        good_angles = self.lidar_angles[useful_idxs]

        #correct angles with lidar angle...
        # 1. from lidar distances to points in the LiDAR frame
        pts_x, pts_y = np.cos(good_angles)*good_lidar, np.sin(good_angles)*good_lidar
        pts_lidar_frame = np.vstack((pts_x, pts_y))
        pts_lidar_frame_stacked = np.vstack((pts_lidar_frame, np.zeros(pts_lidar_frame.shape[1])))
        pts_lidar_frame_stacked = np.vstack((pts_lidar_frame_stacked, np.ones(pts_lidar_frame.shape[1])))

        # 2. from LiDAR frame to the body frame
        v = np.array([0,0,self.lidar_height])

        T_hb = euler_to_se3(0,head_angle,neck_angle,v) #Transformation from lidar to body frame

        # 3. from body frame to world frame
        body_world_translation = np.array([p[0], p[1],self.head_height])
        T_bw = euler_to_se3(head_pose[0],head_pose[1],head_pose[2],body_world_translation) #Transformation from body to world frame
        T_total = T_bw@T_hb
        pts_world = T_total@pts_lidar_frame_stacked #pre or post multiply???
        not_ground_contacts = np.where(pts_world[2] > 0.1) #idea from Tianhong's post on Piazza
        pts_world = pts_world[:,not_ground_contacts].squeeze()
        return pts_world[0:3]

    def get_control(self, t, step):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. 
        need to use the smart_minus_2d function to get the 
        difference of the two poses and we will simply set 
        this to be the control (delta x, delta y, delta theta)
        """
        if t == 0:
            return np.zeros(3)
        p2 = self.lidar[t]["xyth"]
        p2[2] = self.lidar[t]["rpy"][2]
        p1 = self.lidar[t-step]["xyth"]
        p1[2] = self.lidar[t-step]["rpy"][2]
        control = smart_minus_2d(p2, p1)
        return control

    def dynamics_step(self, t, step=1):
        """"
        Compute the control using get_control and perform 
        that control on each particle to get the updated locations of 
        the particles in the particle filter, remember to add 
        noise using the smart_plus_2d function to each particle.
        """
        mean = np.zeros((3))
        noise = np.random.multivariate_normal(mean, self.Q, self.p.shape[1])
        control = self.get_control(t, step)
        for i in range(self.p.shape[1]):
            p = self.p[:,i]
            p_new = smart_plus_2d(p, control)
            p_new = smart_plus_2d(p_new,noise[i])
            self.p[:,i] = p_new

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """

        lw = np.log(w)+obs_logp #multiplication in log is same as adding two logs
        w = np.exp(lw-slam_t.log_sum_exp(lw))
        assert np.allclose(np.sum(w), 1)
        return w

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def observation_step(self, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
        """
        t_lidar = self.lidar[t]["t"]
        T = self.find_joint_t_idx_from_lidar(t_lidar)
        neck_angle = self.joint["head_angles"][0][T]#first row is neck angles; 
        head_angle = self.joint["head_angles"][1][T]#second row is head angleself. use T to get proper data.
        lidar_data = self.lidar[t]
        head_pose = self.lidar[t]["rpy"]
        logP = np.zeros((self.p.shape[1]))
        pDict = {}
        for i in range(self.p.shape[1]):
            p = self.p[:,i]
            #Can probably preload lots of stuff in rays to world. The only thing that really changes is the last step;
            #the body-->world transformation.
            pts_world = self.rays2world(p, lidar_data["scan"], head_pose, head_angle, neck_angle)
            #use most likely particle as "true location" of robot". Use this to update the map
            prop_idxs = self.map.grid_cell_from_xy(pts_world[0], pts_world[1]).astype(int)
            pDict[i] = prop_idxs
            total_ident = self.map.cells[prop_idxs[1],prop_idxs[0]]
            logP[i] = np.sum(total_ident)
        self.w = self.update_weights(self.w, logP)
        chosen_idx = np.argmax(self.w)
        select_idxs = pDict[chosen_idx]
        mask = np.zeros((self.map.cells.shape)).astype(int)
        mask[select_idxs[1], select_idxs[0]] = 1
        self.map.log_odds[np.where(mask)]+= 1.5*self.lidar_log_odds_occ

        contourMask = np.zeros((self.map.cells.shape))
        if self.p.shape[1] > 1:
            pose_in_grid = self.map.grid_cell_from_xy(self.p[0, chosen_idx], self.p[1,chosen_idx], indiv=True)
        else:
            pose_in_grid = self.map.grid_cell_from_xy(self.p[0], self.p[1], indiv=True)
        contours = np.hstack((select_idxs, pose_in_grid)).astype(int) #x and y are already flipped, so no need to flip further.

        cv2.drawContours(contourMask, np.array([contours.T]), -1, color = self.lidar_log_odds_free*0.5, thickness = -1)
        self.map.log_odds += contourMask

        self.map.log_odds = np.clip(self.map.log_odds, -self.map.log_odds_max, self.map.log_odds_max)
        new_occupied = np.where(self.map.log_odds >= self.map.log_odds_thresh)
        new_empty = np.where(self.map.log_odds < self.map.log_odds_thresh)
        self.map.cells[new_occupied] = 1
        self.map.cells[new_empty] = 0
        self.resample_particles()


    def resample_particles(self):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(self.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/self.n < self.resampling_threshold:
            self.p, self.w = self.stratified_resampling(self.p, self.w)
            logging.debug('> Resampling')