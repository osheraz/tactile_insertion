#!/usr/bin/python

import rospy
import copy
import sys
import numpy as np
import tf
from geometry_msgs.msg import Point  # or Pose, if orientation is needed


def avoid_jumps(q_Current, q_Prev):
    q_Current = np.array(q_Current)
    q_Prev = np.array(q_Prev)

    # norm_diff = np.linalg.norm(q_Prev - q_Current) ** 2
    # norm_sum = np.linalg.norm(q_Prev + q_Current) ** 2

    # if norm_diff < norm_sum:

    if np.sign(q_Current[0]) != np.sign(q_Prev[0]):
        return -q_Current
    else:
        return q_Current


def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0, -omg[2], omg[1]],
                     [omg[2], 0, -omg[0]],
                     [-omg[1], omg[0], 0]])


def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]


def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
    np.c_[np.dot(VecToso3(p), R), R]]


from scipy.spatial.transform import Rotation


def average_quaternions(rotations):
    # Convert rotations to quaternions
    quaternions = [Rotation.from_quat(rot).as_quat() for rot in rotations]

    # Convert quaternions to rotation matrices
    matrices = [Rotation.from_quat(q).as_matrix() for q in quaternions]

    # Average rotation matrices
    average_matrix = np.mean(matrices, axis=0)

    # Convert the average matrix back to a quaternion
    average_quaternion = Rotation.from_matrix(average_matrix).as_quat()

    return average_quaternion


class Test:

    def __init__(self):
        """
        Just wrapping everything cuz melodic don't like python3+
        """
        self.joints = None
        self.pose = None
        self.rot = None
        self.tl = tf.TransformListener()
        self.socket_pos_pub = rospy.Publisher('socket_position', Point, queue_size=5)

    def tf_trans(self, target_frame, source_frame):
        try:
            # listen to transform, from source to target. if source is 0 and target is 1 than A_0^1
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None

    def calc_socket_pos(self):

        """
        Overwrite but keep syncing the update
        manipulator api publisher is w.r.t link7
        our publisher is w.r.t to eef.
        """
        try:
            trans_top_left, rot_top_left = self.tf_trans('/base_link', '/top_left')
            trans_bottom_left, rot_bottom_left = self.tf_trans('/base_link', '/bottom_left')
            trans_top_right, rot_top_right = self.tf_trans('/base_link', '/top_right')
            trans_bottom_right, rot_bottom_right = self.tf_trans('/base_link', '/bottom_right')
            # circle_pos, circle_rot = self.tf_trans('/base_link', '/circle')

            # Average x, y, and z coordinates of the four corners
            center_x = (trans_top_left[0] + trans_bottom_left[0] + trans_top_right[0] + trans_bottom_right[0]) / 4.0
            center_y = (trans_top_left[1] + trans_bottom_left[1] + trans_top_right[1] + trans_bottom_right[1]) / 4.0
            center_z = (trans_top_left[2] + trans_bottom_left[2] + trans_top_right[2] + trans_bottom_right[2]) / 4.0

            print('pos', [center_x, center_y, center_z])
            # print(circle_pos)
            # Create a message object for the position
            socket_pos_msg = Point()
            socket_pos_msg.x = center_x
            socket_pos_msg.y = center_y
            socket_pos_msg.z = center_z

            # Publish the message
            self.socket_pos_pub.publish(socket_pos_msg)

            orientations = [rot_top_left, rot_bottom_left, rot_top_right, rot_bottom_right]
            center_orientation = average_quaternions(orientations)
            # print('quat', center_orientation)
        except:
            return 'couldnt find mat', None


if __name__ == "__main__":

    rospy.init_node('socket', anonymous=True)

    rate = rospy.Rate(20)

    test = Test()

    rospy.logwarn('[arm_control] node is ready')

    while not rospy.is_shutdown():
        test.calc_socket_pos()
