
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected
        self.previous_index = 0


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False
        frame_height_from_bottom = 5
        frame_height, frame_width = cv_image.shape[:2]  # [:2] extracts height and width

        # convert frame to grayscale more effectively compare color differences between pixels
        gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        analyzed_row = gray_frame[frame_height - frame_height_from_bottom]

        mean_value = np.mean(analyzed_row)

        middle_list = []
        for i in range(analyzed_row.size):
            # introducing a threshold properly identifies frames with no road
            if analyzed_row[i] < mean_value - 10:
                middle_list.append(i)

        # Convert the list to a NumPy array
        middle_array: np.ndarray = np.array(middle_list)

        # If no road is detected, the ball stays in it's last position
        if middle_array.size == 0:
            ball_center_index = self.previous_index
            if self.timeout >= 5:
                done = True
            else:
                self.timeout += 1
        else:
            ball_center_index = np.mean(middle_array)
            self.previous_index = ball_center_index
            self.timeout = 0

        center_pt = (int(ball_center_index), frame_height - frame_height_from_bottom) #(x, y)
        radius = 10
        color = (255, 0, 255)
        line_thickness = -1

        section_number = (int)(ball_center_index * 10 / frame_width)
        state[section_number] = 1

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        color = (0, 255, 0) 
        thickness = 3
        line_type = cv2.LINE_AA

        # Put the number on the ball's position
        cv2.putText(cv_image, str(section_number), center_pt, font, font_scale, color, thickness, line_type)

        cv2.circle(cv_image, center_pt, radius, color, line_thickness)
        
        # cv2.imshow("camera feed", cv_image)
        # cv2.waitKey(1)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
                if state == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] or state == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
                    reward = 1000
                    print("went straight good")
            elif action == 1:  # LEFT
                reward = 2
                if state == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                    reward = 50
                    print("good left")
            else:
                reward = 2  # RIGHT
                if state == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
                    reward = 50
                    print("good right")
        else:
            reward = -2000

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
