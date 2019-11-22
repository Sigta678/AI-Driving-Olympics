import vrep
import time
import pygame

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

#EP_MAX = 1000
#EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
#BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10

#S_DIM = image_shrinkage_feature + 3(v_x,v_y), get rid of w
#S_DIM = 64*64*3+2 # For test
S_DIM = 2 #Don't take too complicated things into consideration, we need to make it an ablation study
A_DIM = 2

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),
    dict(name='clip', epsilon=0.2),
][1]

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :] # s:(image_size) -> s:(1,image_size), to fit the network
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :] # Just make s from 1-dim to 2-dim as (1, s.shape[0])
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

class Agent():

    def __init__(self, robot):
        # Attach robot
        self.robot = robot

    def update(self):
        # Do something here
        # Make smart moves
        pass

    def getRobotPosition(self):
        return self.robot.getPosition()


class VisualAgent(Agent):

    def __init__(self, robot):
        Agent.__init__(self, robot)
        self.game_finished = 0
        self.last_time = -1
        self.wait_time = 1
        self.ppo = PPO()
        self.all_ep_r = []
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
        self.ep_r = 0
        self.s = []
        self.robot.setWheelVelocity(2, 2)  # Init velocity

    def get_corner(self):
        _, sensorImage = self.robot.getVisionSensor()
        print(sensorImage)
        gray = cv2.cvtColor(sensorImage, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(src=gray, blockSize=8, ksize=23, k=0.04)
        corner_test_flat = np.argmax(dst.flatten()) # Too wierd, just get the most highest value for one corner
        corner_test_coor_x = corner_test_flat % 512
        corner_test_coor_y = 512 - int(corner_test_flat / 512)
        corner_test = []
        corner_test.append(corner_test_coor_x)
        corner_test.append(corner_test_coor_y)
        corner_test = np.array(corner_test)
        print(corner_test)
        return corner_test

    def update(self, sumoAgent):
        # Pass the value inside the update
        winner = sumoAgent.getWinner()
        # Not ended for an episode
        if winner == -1:
            a = self.robot.getWheelVelocity()  # [right_velocity, left_velocity]

            s = self.get_corner()
            
            a = self.ppo.choose_action(s)

            left_w = a[0]
            right_w = a[1]
            self.robot.setWheelVelocity(left_w, right_w) #Limitation is [2,-2]
            self.last_time = time.time()

            v_, w_ = self.robot.getVelocity()

            s_ = self.get_corner()
            
            winner_ = sumoAgent.getWinner()

            if winner_ == -1:
                # image_center = left_corner - right_corner / 2
                # robot_center = object_top_left_corner - object_low_right_corner / 2 #[x_mid,y_mid]
                #image_center = [200, 200]
                #robot_center = np.random.randint(100, size=2) + [50, 50]  # A shift

                #dist = np.sqrt(image_center - robot_center)
                # r = -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2) # https://github.com/openai/gym/wiki/Pendulum-v0
                #r = (0.1 * (1 / (
                #            10 * dist)))  # Because the game is not ended, https://github.com/openai/gym/wiki/CartPole-v0
                r = 0
            elif winner_ == 1:
                r = 1  # Robot1 wins(Robot1's view)
            elif winner_ == 2:
                r = -1  # Robot2 wins(Robot1's view)

            self.buffer_s.append(s)
            self.buffer_a.append(a)
            self.s = s_
            self.ep_r += r

        else:
            # Update PPO

            # Need to revise ppo.get_v(s_), the s_ dimension processing
            v_s_ = self.ppo.get_v(self.s)

            discounted_r = []  # All 1's, something could discount
            for r in self.buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(self.buffer_s), np.vstack(self.buffer_a), np.array(discounted_r)[:, np.newaxis]
            self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
            self.ppo.update(bs, ba, br)

            if self.game_finished == 0:
                self.all_ep_r.append(self.ep_r)
            else:
                self.all_ep_r.append(self.all_ep_r[-1] * 0.9 + self.ep_r * 0.1)

            self.game_finished += 1
            print('Ep: %i' % self.game_finished, "|Ep_r: %i" % self.ep_r,
                  ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',)
            self.ep_r = 0

class StayOnAgent(Agent):

    def __init__(self, robot):
        Agent.__init__(self, robot)
        self.last_time = -1
        self.wait_time = 1

    def update(self, sumoAgent):
        # Check line sensor
        leftReading, rightReading = self.robot.getLineSensors()
        
        _, sensorImage = self.robot.getVisionSensor()
        # print(sensorImage)
        # detector = cv2.SimpleBlobDetector_create()
        # keypoints = detector.detect(sensorImage)
        gray = cv2.cvtColor(sensorImage, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)

        #dst = cv2.cornerHarris(src=gray, blockSize=8, ksize=23, k=0.04)
        dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
        #print(dst.shape) #Harris value for all the pixel, a lot of large value set
        
        sensorImage[dst > 0.01 * dst.max()] = [0, 0, 255]
        #corner_list = sensorImage[dst > 0.01 * dst.max()] #dst is the value, we are going to compare between pixel index
        
        #print(dst.shape)
        print(sensorImage.shape)
        
        #sensorImage_list = sensorImage.flatten()
        #corner_list_value = gray[dst > 0.01 * dst.max()] # The value, not coordinate
        #corner_list_value = list(sensorImage_list).index(255)
        
        corner_list_index = []
        index = 0
        #print(list(dst))
        dst_flat = dst.flatten()

        for value in list(dst_flat):
            if value > 0.3*dst.max():
                pixel_row = index%32
                pixel_col = 32 - int(index / 32)
                #sensorImageChannel = sensorImage[pixel_row-1, pixel_col-1, :]
                #print(sensorImageChannel)
                #if sensorImageChannel[0] == sensorImageChannel[1] and sensorImageChannel[0] == sensorImageChannel[2] and sensorImageChannel[0]==0:
                #    continue
                #else:
                corner_list_index.append([pixel_row,pixel_col])
            index += 1
        #corner_list_value = [index for value in list(sensorImage_list) if value == 255]
        #print(corner_list_index)
        mean_feature_coordinate = np.mean(corner_list_index,axis=0)
        print(mean_feature_coordinate)
        #corner_top = np.argmin(corner_list, axis = 0)
        #print(corner_top)
        #corner_bottom = np.argmax(corner_list, axis = 0)
        #print(corner_bottom)
        '''
        corner_test_flat = np.argmax(dst.flatten()) # Too wierd, just get the most highest value for one corner
        corner_test_coor_x = corner_test_flat % 512
        corner_test_coor_y = 512 - int(corner_test_flat / 512)
        corner_test = [corner_test_coor_x, corner_test_coor_y]
        print(corner_test)
        '''
        #corner_bottom = [max(corner_list[:,0]),max(corner_list[:,1])]
        #sensorImage[dst==corner_top] = [0,0,255]
        
        #corner_avg = np.average(corner_list,axis=0)
        #sensorImage[dst==corner_avg] = [0,0,255]
        #print(corner_avg)
        
        #print(sensorImage.shape)
        #cv2.imshow('dst', sensorImage)
        #if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()
        print("Mean coordinate:",mean_feature_coordinate)
        #if mean_feature_coordinate[0] <5 and mean_feature_coordinate[1] < 5:
        #if np.isnan(list(mean_feature_coordinate)):
        if len(corner_list_index)==0 or mean_feature_coordinate[0] < 5:
            #self.robot.setWheelVelocity(-2.0, -2.0)
            self.last_time = time.time()
        elif mean_feature_coordinate[0] < 16 and mean_feature_coordinate[0] > 10:
            self.robot.setWheelVelocity(2.0, 2.0)
            self.last_time = time.time()
        elif mean_feature_coordinate[0] > 16:
            if mean_feature_coordinate[1] > 16:
                self.robot.setWheelVelocity(2.0, 0.0)
                self.last_time = time.time()
            elif mean_feature_coordinate[1] < 16 and mean_feature_coordinate[1] > 10:
                self.robot.setWheelVelocity(0.0, 2.0)
                self.last_time = time.time()
            #elif mean_feature_coordinate
        
        '''
        # Back up straight
        if leftReading and rightReading:
            self.robot.setWheelVelocity(-2.0, -2.0)
            self.last_time = time.time()
        # Back up to right
        elif leftReading:
            self.robot.setWheelVelocity(-2.0, 0.0)
            self.last_time = time.time()
        # Back up to left
        elif rightReading:
            self.robot.setWheelVelocity(0.0, -2.0)
            self.last_time = time.time()
        # Go forward
        else:
            if time.time() - self.last_time > self.wait_time:
                self.robot.setWheelVelocity(2.0, 2.0)
        '''
# https://www.pygame.org/docs/ref/key.html
class HumanAgent(Agent):

    def __init__(self, robot):
        # Initialize robot and pygame
        Agent.__init__(self, robot)
        pygame.init()
        pygame.display.set_mode((640, 480), 0, 0)
        # Counter for print
        self.count = 0

    def __del__(self):
        pygame.quit()

    def update(self):
        # Get key events
        pygame.event.get()
        left_w = 2.0
        right_w = 0.0
        # Check all pressed keys
        pressed = pygame.key.get_pressed()
        # Update velocity
        if pressed[pygame.K_w]:
            left_w += 2.0
            right_w += 2.0
        if pressed[pygame.K_a]:
            left_w -= 2.0
            right_w += 2.0
        if pressed[pygame.K_d]:
            left_w += 2.0
            right_w -= 2.0
        if pressed[pygame.K_s]:
            left_w -= 2.0
            right_w -= 2.0
        if pressed[pygame.K_ESCAPE]:
            pygame.quit()
        # Set velocity
        self.robot.setWheelVelocity(left_w, right_w)
        self.count += 1

        if(self.count % 5 == 0):
            v,w = self.robot.getVelocity()

            print("Linear v: " + str(v))
            print("Angualr w: " + str(w))
            print("")
