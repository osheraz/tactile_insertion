#!/usr/bin/python 

'''
----------------------------
Author: Osher Azulay
Date: October 2018
----------------------------
'''

import rospy
import numpy as np
from std_msgs.msg import Float64, String
from std_msgs.msg import Float64MultiArray, Float32MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, EmptyResponse
import tf
import numpy
import rospy
import time
import tf
import math
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Char
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Float64
import sys
import geometry_msgs.msg
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String, Float32MultiArray, Bool, Int16
from std_srvs.srv import Empty, EmptyResponse, SetBool
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3, WrenchStamped, \
    Wrench, PoseWithCovariance, Pose
from scipy.spatial.transform import Rotation as R
from tf.transformations import translation_matrix, rotation_matrix, translation_from_matrix, rotation_from_matrix, \
    concatenate_matrices

origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)


class Tracker():
    '''
    fuse data by subscribing to tag detection topics
    '''

    def __init__(self):
        rospy.init_node('april_tracker_multiple_camera_fusion', anonymous=True)
        self._setup_workspace()
        rate = rospy.Rate(100)

        rospy.Subscriber('/test1/tag_detections', AprilTagDetectionArray, self.callbackDetection1)
        rospy.Subscriber('/test2/tag_detections', AprilTagDetectionArray, self.callbackDetection2)
        rospy.Subscriber('/grasped_object_id', Int16, self.callbackUpdateId)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        rospy.loginfo("Setup apriltag marker detection with multiple cameras:")

        while not rospy.is_shutdown():
            self._fuse()
            rate.sleep()

    def _setup_workspace(self):
        # TODO: convert to read from yaml file
        self.object_map = {1: 'circle', 2: 'hexagon', 3: 'ellipse', 4: 'hole'}
        self.objects_pose_map_camera1 = {'circle': [Pose(), 0.0], 'hexagon': [Pose(), 0.0], 'ellipse': [Pose(), 0.0],
                                         'hole': [Pose(), 0.0]}
        self.objects_pose_map_camera2 = {'circle': [Pose(), 0.0], 'hexagon': [Pose(), 0.0], 'ellipse': [Pose(), 0.0],
                                         'hole': [Pose(), 0.0]}
        self.fused_map = {key: Pose() for key in self.objects_pose_map_camera1.keys()}
        self.startup_time = rospy.get_time()

    def _fuse(self):

        # latest = rospy.Time(0)
        # tf_exceptions = (tf.LookupException,
        #                  tf.ConnectivityException,
        #                  tf.ExtrapolationException)
        #
        # try:
        #     (trans_camera1, rot_camera1) = self.tf_listener.lookupTransform('world', 'camera1_link', latest)
        # except tf_exceptions:
        #     rospy.logwarn("No transformation from %s to %s" % ('world', 'camera1_link'))
        #     return False
        # try:
        #     (trans_camera2, rot_camera2) = self.tf_listener.lookupTransform('world', 'camera2_link', latest)
        # except tf_exceptions:
        #     rospy.logwarn("No transformation from %s to %s" % ('world', 'camera2_link'))
        #     return False

        cur_t = rospy.get_time() - self.startup_time
        delta_update = 0.05

        for key in self.fused_map.keys():

            last_time_update_cam1 = self.objects_pose_map_camera1[key][1]
            last_time_update_cam2 = self.objects_pose_map_camera2[key][1]

            if abs(last_time_update_cam1 - cur_t) < delta_update and abs(last_time_update_cam2 - cur_t) < delta_update:
                pose_from_camera_1 = self.objects_pose_map_camera1[key][0]
                pose_from_camera_2 = self.objects_pose_map_camera2[key][0]

                x1, y1, z1 = pose_from_camera_1.position.x, pose_from_camera_1.position.y, pose_from_camera_1.position.z
                q_x1, q_y1, q_z1, q_w1 = pose_from_camera_1.orientation.x, pose_from_camera_1.orientation.y, \
                                         pose_from_camera_1.orientation.z, pose_from_camera_1.orientation.w
                x2, y2, z2 = pose_from_camera_2.position.x, pose_from_camera_2.position.y, pose_from_camera_2.position.z
                q_x2, q_y2, q_z2, q_w2 = pose_from_camera_2.orientation.x, pose_from_camera_2.orientation.y, \
                                         pose_from_camera_2.orientation.z, pose_from_camera_2.orientation.w

                position = ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
                orientation = ((q_x1 + q_x2) / 2, (q_y1 + q_y2) / 2, (q_z1 + q_z2) / 2, (q_w1 + q_w2) / 2)
                self.pub_tf(key, position, orientation)
                print(1)

            elif abs(last_time_update_cam1 - cur_t) < delta_update and abs(last_time_update_cam2 - cur_t) > delta_update:
                pose_from_camera_1 = self.objects_pose_map_camera1[key][0]
                x1, y1, z1 = pose_from_camera_1.position.x, pose_from_camera_1.position.y, pose_from_camera_1.position.z
                q_x1, q_y1, q_z1, q_w1 = pose_from_camera_1.orientation.x, pose_from_camera_1.orientation.y, \
                                         pose_from_camera_1.orientation.z, pose_from_camera_1.orientation.w
                position = (x1, y1, z1)
                orientation = (q_x1, q_y1, q_z1, q_w1)
                self.pub_tf(key, position, orientation)
                print(2)

            elif abs(last_time_update_cam1 - cur_t) > delta_update and abs(last_time_update_cam2 - cur_t) < delta_update:
                pose_from_camera_2 = self.objects_pose_map_camera2[key][0]
                x2, y2, z2 = pose_from_camera_2.position.x, pose_from_camera_2.position.y, pose_from_camera_2.position.z
                q_x2, q_y2, q_z2, q_w2 = pose_from_camera_2.orientation.x, pose_from_camera_2.orientation.y, \
                                         pose_from_camera_2.orientation.z, pose_from_camera_2.orientation.w
                position = (x2, y2, z2)
                orientation = (q_x2, q_y2, q_z2, q_w2)

                self.pub_tf(key, position, orientation)
                print(3)


    def pub_tf(self, key, pos, ori):

        obj_pos = pos
        obj_quat = ori
        print(pos)
        self.tf_broadcaster.sendTransform(obj_pos,
                                          obj_quat,
                                          rospy.Time.now(),
                                          str(key),
                                          "world")

    def callbackDetection1(self, msg):

        detection_array = msg.detections
        for detect in detection_array:
            tag_id = detect.id[0]
            if tag_id not in self.object_map.keys(): continue
            which_object = self.object_map[tag_id]
            t = rospy.get_time() - self.startup_time

            ps = PoseStamped()
            ps.header = detect.pose.header
            ps.pose = detect.pose.pose.pose
            p = self.tf_listener.transformPose('world', ps)  # with respect to the world

            pose = p.pose
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            dist = np.linalg.norm([x, y, z])
            if dist > 1.0:  # (m)
                rospy.logwarn("[camera1] Tag of %s out of range (dist=%f)" % (which_object, dist))
            self.objects_pose_map_camera1[which_object][0] = pose
            self.objects_pose_map_camera1[which_object][1] = t

    def callbackDetection2(self, msg):

        detection_array = msg.detections
        for detect in detection_array:
            tag_id = detect.id[0]
            if tag_id not in self.object_map.keys(): continue
            which_object = self.object_map[tag_id]
            t = rospy.get_time() - self.startup_time

            ps = PoseStamped()
            ps.header = detect.pose.header
            ps.pose = detect.pose.pose.pose

            p = self.tf_listener.transformPose('world', ps)  # with respect to the world

            pose = p.pose
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            dist = np.linalg.norm([x, y, z])
            if dist > 1.0:  # (m)
                rospy.logwarn("[camera1] Tag of %s out of range (dist=%f)" % (which_object, dist))
            self.objects_pose_map_camera2[which_object][0] = pose
            self.objects_pose_map_camera2[which_object][1] = t

    def callbackUpdateId(self, msg):
        self.grasped_object_id = msg.data


class TrackerByTf():
    '''
    fuse data by directly subscribing to tf topics
    '''

    def __init__(self):
        rospy.init_node('april_tracker_multiple_camera_fusion', anonymous=True)
        self._setup_workspace()
        rate = rospy.Rate(100)

        rospy.Subscriber('/grasped_object_id', Int16, self.callbackUpdateId)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        rospy.loginfo("Setup apriltag marker detection with multiple cameras:")

        while not rospy.is_shutdown():
            self._fuse()
            rate.sleep()

    def _setup_workspace(self):
        # TODO: convert to read from yaml file
        self.object_map = {1: 'circle', 2: 'hexagon', 3: 'ellipse', 4: 'hole'}
        self.objects_pose_map_camera1 = {'circle': [Pose(), 0.0], 'hexagon': [Pose(), 0.0], 'ellipse': [Pose(), 0.0],
                                         'hole': [Pose(), 0.0]}
        self.objects_pose_map_camera2 = {'circle': [Pose(), 0.0], 'hexagon': [Pose(), 0.0], 'ellipse': [Pose(), 0.0],
                                         'hole': [Pose(), 0.0]}
        self.fused_map = {key: None for key in self.objects_pose_map_camera1.keys()}
        self.startup_time = rospy.get_time()

    def _fuse(self):
        # rospy.loginfo(self.objects_pose_map_camera1)
        # rospy.loginfo(self.objects_pose_map_camera2)
        # TF Tree transformations.
        latest = rospy.Time(0)
        tf_exceptions = (tf.LookupException,
                         tf.ConnectivityException,
                         tf.ExtrapolationException)
        # Transform to world frame - Publish transform and listen to
        # transformation to world frame.
        try:
            (trans_camera1, rot_camera1) = self.tf_listener.lookupTransform('world', 'camera1_link', latest)
        except tf_exceptions:
            rospy.logwarn("No transformation from %s to %s" % ('world', 'camera1_link'))
            return False
        try:
            (trans_camera2, rot_camera2) = self.tf_listener.lookupTransform('world', 'camera2_link', latest)
        except tf_exceptions:
            rospy.logwarn("No transformation from %s to %s" % ('world', 'camera2_link'))
            return False

        cur_t = rospy.get_time()
        delta_update = 0.1

        for key in self.fused_map.keys():

            last_time_update_cam1 = self.objects_pose_map_camera1[key][1]
            last_time_update_cam2 = self.objects_pose_map_camera2[key][1]

            if abs(last_time_update_cam1 - cur_t) < delta_update and abs(last_time_update_cam2 - cur_t) < delta_update:
                pose_from_camera_1 = self.objects_pose_map_camera1[key][0]
                pose__fromcamera_2 = self.objects_pose_map_camera2[key][0]

        # TF Tree transformations.
        # latest = rospy.Time(0)
        # tf_exceptions = (tf.LookupException,
        #                  tf.ConnectivityException,
        #                  tf.ExtrapolationException)
        # Transform to world frame - Publish transform and listen to
        # transformation to world frame.
        # try:
        #     (trans, rot) = self.tf_listener.lookupTransform(
        #         world_frame, self.vehicle_frame, latest)
        #     # Add estimate to pose estimates.
        #     euler = euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
        #     pose_estimates.append(np.array([trans[0], trans[1], euler[2]]))
        # except tf_exceptions:
        #     rospy.logwarn("No transformation from %s to %s" %
        #                   (world_frame, self.vehicle_frame))
        #     return False

    def callbackUpdateId(self, msg):
        self.grasped_object_id = msg.data


if __name__ == '__main__':

    try:
        Tracker()
    except rospy.ROSInterruptException:
        pass
