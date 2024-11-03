#!/usr/bin/env python

# Author: Osher Azulay

import os
import sys
import time
import rospy
from scipy.spatial.transform import Rotation as R
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse
from moveit_commander.conversions import pose_to_list
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryAction, JointTrajectoryGoal, FollowJointTrajectoryAction, \
    FollowJointTrajectoryGoal
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Time
from hand_control.srv import observation, IsDropped, TargetAngles
from hand_control.srv import RegraspObject, close, planroll, TargetPos
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, GetLinkState, GetLinkStateRequest
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from std_msgs.msg import Header, Float32MultiArray
from std_msgs.msg import Duration
from std_srvs.srv import Empty
from geometry_msgs.msg import WrenchStamped

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
import tf
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker
from tf.transformations import translation_matrix, rotation_matrix, translation_from_matrix, rotation_from_matrix
import matplotlib.pyplot as plt
from robotiq_force_torque_sensor.srv import sensor_accessor

class MoveItController(object):

    def __init__(self, verbose):

        # Initialize the node
        super(MoveItController, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        self.verbose = verbose
        rospy.init_node('move_it_arm_controller')

        try:
            self.with_gripper = rospy.get_param(rospy.get_namespace() + "~with_gripper", False)
            if self.with_gripper:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 6)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(
                rospy.get_namespace() + 'move_group/display_planned_path',
                moveit_msgs.msg.DisplayTrajectory,
                queue_size=20)
            if self.with_gripper:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
            self.is_init_success = True
        except Exception as e:
            print (e)
            self.is_init_success = False

        else:
            self.is_init_success = True

        # Misc variables
        self.object_name = ''
        self.planning_frame = self.arm_group.get_planning_frame()
        self.eef_link = self.arm_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        # self.get_planning_feedback()

    def define_workspace(self):
        # Walls are defined with respect to the coordinate frame of the robot base, with directions
        # corresponding to standing behind the robot and facing into the table.
        rospy.sleep(0.6)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'

        # self.robot.get_planning_frame()
        table_pose = PoseStamped()
        table_pose.header = header
        table_pose.pose.position.x = 0
        table_pose.pose.position.y = 0
        table_pose.pose.position.z = -0.0001
        self.scene.remove_world_object('bottom')
        self.scene.add_plane(name='bottom', pose=table_pose, normal=(0, 0, 1))

        upper_pose = PoseStamped()
        upper_pose.header = header
        upper_pose.pose.position.x = 0
        upper_pose.pose.position.y = 0
        upper_pose.pose.position.z = 0.6
        self.scene.remove_world_object('upper')
        self.scene.add_plane(name='upper', pose=upper_pose, normal=(0, 0, 1))

        back_pose = PoseStamped()
        back_pose.header = header
        back_pose.pose.position.x = 0
        back_pose.pose.position.y = -0.4 # -0.25
        back_pose.pose.position.z = 0
        self.scene.remove_world_object('rightWall')
        self.scene.add_plane(name='rightWall', pose=back_pose, normal=(0, 1, 0))

        front_pose = PoseStamped()
        front_pose.header = header
        front_pose.pose.position.x = -0.25
        front_pose.pose.position.y = 0.0  # 0.52 # Optimized (0.55 NG)
        front_pose.pose.position.z = 0
        self.scene.remove_world_object('backWall')
        # self.scene.add_plane(name='backWall', pose=front_pose, normal=(1, 0, 0))

        right_pose = PoseStamped()
        right_pose.header = header
        right_pose.pose.position.x = 0.6  # 0.2
        right_pose.pose.position.y = 0
        right_pose.pose.position.z = 0
        self.scene.remove_world_object('frontWall')
        self.scene.add_plane(name='frontWall', pose=right_pose, normal=(1, 0, 0))

        left_pose = PoseStamped()
        left_pose.header = header
        left_pose.pose.position.x = 0.0  # -0.54
        left_pose.pose.position.y = 0.4
        left_pose.pose.position.z = 0
        self.scene.remove_world_object('leftWall')
        self.scene.add_plane(name='leftWall', pose=left_pose, normal=(0, 1, 0))
        rospy.sleep(0.6)

    def all_close(self, goal, actual, tolerance):
        """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
        all_equal = True
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            return self.all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

        return True

    def get_planning_feedback(self):
        planning_frame = self.arm_group.get_planning_frame()
        print ("============ Planning frame: %s" % planning_frame)

        # print the name of the end-effector link for this group:
        eef_link = self.arm_group.get_end_effector_link()
        print ("============ End effector link: %s" % eef_link)

        # get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print ("============ Available Planning Groups:", group_names)

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print ("============ Printing robot state")
        print (self.robot.get_current_state())
        print ("")

    def reach_named_position(self, target):
        arm_group = self.arm_group

        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)
        # Plan the trajectory
        # planned_path1 = arm_group.plan()
        # Execute the trajectory and block while it's not finished
        # arm_group.execute(planned_path1, wait=True)
        return arm_group.go(wait=True)

    def reach_joint_angles(self, target, tolerance):

        success = True
        joint_positions = self.arm_group.get_current_joint_values()

        # Get the current joint positions
        if self.verbose:
            rospy.loginfo("Printing current joint positions before movement :")
            for p in joint_positions: rospy.loginfo(p)

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the joint target configuration
        if len(target) != self.degrees_of_freedom:
            print('Dimensions error')
        else:
            self.arm_group.set_joint_value_target(target)

            # Plan and execute in one command
            success &= self.arm_group.go(wait=True)

            # Show joint positions after movement
            new_joint_positions = self.arm_group.get_current_joint_values()
            if self.verbose:
                rospy.loginfo("Printing current joint positions after movement :")
                for p in new_joint_positions: rospy.loginfo(p)

        return success and self.all_close(target, new_joint_positions, tolerance)

    def get_cartesian_pose(self, verbose=False):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        if verbose:
            rospy.loginfo("Actual cartesian pose is : ")
            rospy.loginfo(pose.pose)

        return pose.pose

    def reach_cartesian_pose(self, pose, tolerance, constraints, wait=True):

        # Set the tolerance
        self.arm_group.set_goal_position_tolerance(tolerance)

        # Set the trajectory constraint if one is specified
        if constraints is not None:
            self.arm_group.set_path_constraints(constraints)

        # Get the current Cartesian Position
        self.arm_group.set_pose_target(pose)

        # Plan and execute
        if self.verbose:
            rospy.loginfo("Planning and going to the Cartesian Pose")

        self.arm_group.go(wait=wait)
        self.arm_group.clear_pose_targets()
        current_pose = self.arm_group.get_current_pose().pose

        return self.all_close(pose, current_pose, 0.01)

    def reach_gripper_position(self, relative_position):

        # We only have to move this joint because all others are mimic!
        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        try:
            val = gripper_joint.move(
                relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos,
                True)
            return val
        except:
            return False

    def plan_cartesian_path(self, pose, eef_step=0.001):

        waypoints = []
        # start with the current pose
        # waypoints.append(self.arm_group.get_current_pose().pose)

        wpose = self.arm_group.get_current_pose().pose#geometry_msgs.msg.Pose()
        wpose.position.x = pose.position.x
        wpose.position.y = pose.position.y
        wpose.position.z = pose.position.z
        # wpose.orientation.x = pose.orientation.x
        # wpose.orientation.y = pose.orientation.y
        # wpose.orientation.z = pose.orientation.z
        # wpose.orientation.w = pose.orientation.w

        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = self.arm_group.compute_cartesian_path(
            waypoints,  # waypoints to follow
            eef_step,  # eef_step
            2.0)  # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

    def execute_plan(self, plan, wait=True):
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        self.arm_group.execute(plan, wait=wait)
        self.arm_group.clear_pose_targets()

    def display_trajectory(self, plan):
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory);

        ## END_SUB_TUTORIAL


class BasicMarker():
    """Basic marker class to visualize on Rviz"""

    def __init__(self, frame, ns, unique_id):
        """Constructor of the Basic Marker Class
    Args:
        frame (str): link to which marker is attached
        ns (str): namespace of the marker
        unique_id (int): unique id for the marker
    """
        self._cmap = plt.cm.get_cmap('autumn_r')
        self._norm = plt.Normalize(vmin=0.0, vmax=5.0, clip=True)

        self.marker_object = Marker()
        self.marker_object.type = Marker.ARROW
        self.marker_object.action = Marker.ADD
        self.marker_object.ns = ns
        self.marker_object.id = unique_id
        self.marker_object.header.frame_id = frame
        self.marker_object.frame_locked = True
        self._set_scale()
        self._set_colour()
        self._set_points([[0, 0, 0], [0, 0, 0]])
        self.updated = False

        # If we want it for ever, 0, otherwise seconds before desapearing
        self.marker_object.lifetime = rospy.Duration(50)

    def _set_points(self, points):
        """Set local position of the marker
    Args:
        points (list[list[float]]): [description]
    """
        self.marker_object.points = []
        for [x, y, z] in points:
            point = Point()
            point.x = x
            point.y = y
            point.z = z
            self.marker_object.points.append(point)

    def _set_colour(self, pressure=0.0):
        """Set marker color based on the pressure
    Args:
        pressure (float, optional): Pressure/force on the link. Defaults to 0.0.
    """
        rgba = self._cmap(self._norm(pressure))
        self.marker_object.color.r = rgba[0]
        self.marker_object.color.g = rgba[1]
        self.marker_object.color.b = rgba[2]
        self.marker_object.color.a = rgba[3]

    def _set_scale(self,
                   diameter=0.005,
                   head_diameter=0.008,
                   head_length=0.0):
        """Sets marker scale
    Args:
        diameter (float, optional): Diameter of the arrow. Defaults to 0.005.
        head_diameter (float, optional): Diameter of the arrow head. Defaults to 0.008.
        head_length (float, optional): Length of the arrow head. Defaults to 0.0.
    """
        self.marker_object.scale.x = diameter
        self.marker_object.scale.y = head_diameter
        self.marker_object.scale.z = head_length

    def update_marker(self, p1, p2, pressure):
        """Update marker properties
    Args:
        p1 (list[float]): Start point of the marker arrow
        p2 (list[float]): End point of the marker arrow
        pressure (float): Pressure on the link
    """
        if not self.updated:
            self.marker_object.header.stamp = rospy.get_rostime()
            self._set_points([p1, p2])
            self._set_colour(np.abs(pressure))
            self.updated = True

    def get_marker(self):
        """Return `Marker` object for ROS message
    Returns:
        `Marker`: Marker for publishing to ROS network
    """
        self.updated = False
        return self.marker_object


class ObjectPublisher():

    def __init__(self, ns, object_props):
        """Constructor for ContactsDisplay class
    Args:
        ns (str): namespace of the marker array
        links (list[str]): List of all links of the sensor
    """

        self.x = object_props[0]
        self.y = object_props[1]
        self.r = object_props[2]
        self.h = object_props[3]



        ## Gazebo
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.get_model_state_req = GetModelStateRequest()
        self.get_model_state_req.model_name = 'object'
        self.get_model_state_req.relative_entity_name = 'world'

        self.publisher = rospy.Publisher("normal", MarkerArray, queue_size=1)
        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()

        self.markers = [
            BasicMarker(frame='object', ns=ns, unique_id=i)
            for i in range(200)
        ]
        self.create_cyl_push_points()

    def publish(self):
        """Publishes `MarkerArray` to the ROS network"""
        markerarray_object = MarkerArray()
        markerarray_object.markers = [
            marker.get_marker()
            for marker in self.markers
        ]
        self.publisher.publish(markerarray_object)

    def create_cyl_push_points(self):
        """ Create pushing point around the finger
    Args:
        data (ContactsState): `ContactsState` message from the gazebo
        ii (int): link identifier
    """
        theta = np.linspace(0, 2 * np.pi, 10)
        H = np.linspace(0, self.h, 10)

        counter = 0
        for j, h in enumerate(H):
            for i, q in enumerate(theta):
                B = np.asarray([
                    self.r * np.cos(q),
                    self.r * np.sin(q),
                    h,
                ])
                H = np.asarray([
                    2 * self.r * np.cos(q),
                    2 * self.r * np.sin(q),
                    h,
                ])
                pressure = 1.0
                self.markers[counter].update_marker(H, B, pressure)
                counter += 1

    def get_object_pose(self, sim=True):

        if sim:
            object_state = self.get_model_state_proxy(self.get_model_state_req)
            x = object_state.pose.position.x
            y = object_state.pose.position.y
            z = object_state.pose.position.z
            orientation = object_state.pose.orientation

            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            obj_pos = PoseStamped()
            obj_pos.header = header
            obj_pos.pose.position.x = x
            obj_pos.pose.position.y = y
            obj_pos.pose.position.z = z
            obj_pos.pose.orientation.x = orientation.x
            obj_pos.pose.orientation.y = orientation.y
            obj_pos.pose.orientation.z = orientation.z
            obj_pos.pose.orientation.w = orientation.w

        return obj_pos

def get_ft_reading():
    msg = rospy.wait_for_message('/robotiq_force_torque_wrench', WrenchStamped) # _filtered

    return np.array([msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z])

def rotate_pose_by_rpy(in_pose, roll, pitch, yaw):
  """
  Apply an RPY rotation to a pose in its parent coordinate system.
  """
  try:
    if in_pose.header: # = in_pose is a PoseStamped instead of a Pose.
      in_pose.pose = rotate_pose_by_rpy(in_pose.pose, roll, pitch, yaw)
      return in_pose
  except:
    pass
  q_in = [in_pose.orientation.x, in_pose.orientation.y, in_pose.orientation.z, in_pose.orientation.w]
  q_rot = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
  q_rotated = tf.transformations.quaternion_multiply(q_in, q_rot)

  rotated_pose = copy.deepcopy(in_pose)
  rotated_pose.orientation = geometry_msgs.msg.Quaternion(*q_rotated)
  return rotated_pose

def get_mt_reading():
    msg = rospy.wait_for_message('/gripper/load', Float32MultiArray) # _filtered
    return np.array(msg.data)[1:]

def get_obj_relative_pos():
    msg = rospy.wait_for_message('/hand_control/obj_relative_pos', Float32MultiArray)
    return  np.array(msg.data)

def get_obj_pos():
    msg = rospy.wait_for_message('/hand_control/obj_pos', Float32MultiArray)
    return  np.array(msg.data)

def main():
    #  Init manipulator p arm
    real = True
    with_openhand = True
    add_object_to_scene = False
    define_workspace = False

    if real:
        ft_zero = rospy.ServiceProxy('/robotiq_force_torque_sensor_acc', sensor_accessor)
        ft_zero.call(8, "")

    arm = MoveItController(verbose=False)
    success = arm.is_init_success
    arm.arm_group.set_max_velocity_scaling_factor(1)

    if with_openhand:
        obs_srv = rospy.ServiceProxy('/observation', observation)
        move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        open_srv = rospy.ServiceProxy('/OpenGripper', Empty)
        open_srv()
        close_srv = rospy.ServiceProxy('/CloseGripper', close)
        cur_pose = arm.get_cartesian_pose()

    # # define the workspace around the arm
    if define_workspace: arm.define_workspace()

    # Add finger to the scene
    if add_object_to_scene:
        object_props = [0.0, 0.2, 0.02, 0.15]   # x, y, r, h
        start_h = 0.0
        object_body = geometry_msgs.msg.PoseStamped()
        object_body.header.frame_id = "world"
        object_body.pose.position.x = object_props[0]
        object_body.pose.position.y = object_props[1]
        h = object_props[3]
        object_body.pose.position.z = start_h + h / 2
        arm.scene.add_cylinder("object_body", object_body, h, object_props[2])

        # # Publish pushing points
        # PP = ObjectPublisher(ns='', object_props=object_props)
        # rospy.sleep(3)
        # PP.publish()
        #
        # listener = tf.TransformListener()
        # rospy.sleep(2)
        # object_pose = PP.get_object_pose()

    '''
    test attempt of insert
    '''

    rospy.loginfo('Going to init pose')

    # Moving above the hole
    tolerance = 0.0001 # 1mm
    factor = 0.001
    tp = arm.get_cartesian_pose()
    tp.position.x, tp.position.y, tp.position.z = 0.310970799996, 0.0400073356344, 0.0784212688968 # 0.2
    tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = 0.5, -0.5, 0.5, 0.5
    # success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None)
    # l= 60
    # st = 180
    # rem = np.random.randint(0,2,size=l)
    # cei_rrl[st:st+l] = np.where(cei_rrl[st:st+l], rem, cei_rrl[st:st+l])

    #### Align with object
    tp = arm.get_cartesian_pose()
    obj_pos = get_obj_relative_pos()

    if not np.isnan(np.sum(obj_pos)):
        tp.position.x -= obj_pos[1] + 0.02
        tp.position.y -= obj_pos[0]
        tp.position.z -= obj_pos[2] - 0.07
        tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = 0.5, -0.5, 0.5, 0.5
        success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None)
        close_srv()

    #
    tp = arm.get_cartesian_pose()
    tp.position.z += 0.03
    tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = 0.5, -0.5, 0.5, 0.5
    success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None)


    tp = arm.get_cartesian_pose()
    obj_relative_pos = get_obj_relative_pos()
    obj_pos = get_obj_pos()
    arm.arm_group.set_max_acceleration_scaling_factor(1e-3)
    arm.arm_group.set_max_velocity_scaling_factor(1e-3)
    goal_x = 0.2272448092699051
    goal_y = -0.06375438719987869
    if not np.isnan(np.sum(obj_relative_pos)):
        # added 0.01/-0.05 to approximately center the object
        tp.position.x = goal_x + obj_relative_pos[1]
        tp.position.y = goal_y + obj_relative_pos[0] 
        tp.position.z = 0.05 # - obj_pos[2] + 0.06
        success = arm.reach_cartesian_pose(pose=tp, tolerance=1e-5, constraints=None)



    # Test hysteresis
    hysteresis_test = False
    if hysteresis_test:
        rospy.loginfo('hysteresis , scaling factor ' + str(factor) + ' tolerance: ' + str(tolerance))
        data = []

        for i in range(50):
            x_ = 0.02 #0.0005
            y_ = 0.02 #0.0005
            tp = arm.get_cartesian_pose()
            tp.position.x = 0.287988619809 +  x_ * pow(-1,i)
            tp.position.y = 0.0 + y_ * pow(-1,i)
            tp.position.z = 0.3 # 0.21804222915
            success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None)

            tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = 0.5, -0.5, 0.5, 0.5#0.0, 0.70707811564, 0.0, 0.707135444191

            arm.arm_group.set_max_acceleration_scaling_factor(factor*20)
            arm.arm_group.set_max_velocity_scaling_factor(factor*20)

            success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None)
            arm.arm_group.set_max_acceleration_scaling_factor(factor * 20)
            arm.arm_group.set_max_velocity_scaling_factor(factor * 20)

            theta = np.random.uniform(-np.radians(3),np.radians(3))
            roll,pitch, yaw = tf.transformations.euler_from_quaternion((tp.orientation.x,tp.orientation.y,tp.orientation.z,tp.orientation.w))
            roll += theta
            pitch +=theta
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            tp.orientation.x,tp.orientation.y,tp.orientation.z,tp.orientation.w = quaternion
            success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None)

            p = arm.get_cartesian_pose()
            data.append([p.position.x,p.position.y,p.position.z])
            rospy.loginfo(i)

        data = np.array(data)
        fig, axs = plt.subplots(3)
        axs[0].plot(data[:,0])
        axs[1].plot(data[:,1])
        axs[2].plot(data[:,2])
        plt.show()

    # Test force distribution
    force_layout_test = True
    if force_layout_test:

        # Pressing
        x_ = [ 0.01, 0.0, -0.01, 0.0, 0.01, -0.01, 0.01, -0.01]
        y_ = [ 0.0, 0.01, 0.0, -0.01, 0.01, 0.01, -0.01, -0.01]

        rot_a = 5
        yaw_ =   [-np.radians(rot_a), np.radians(rot_a), 0.,  0., 0.]
        pitch_ = [0.  , 0.,np.radians(rot_a), -np.radians(rot_a), 0.]
        up_to = 30

        for j in range(4):

            combined = []
            combined_mt = []

            for i in range(5):

                # close_srv()

                if real:
                    ft_zero.call(8, "")

                arm.arm_group.set_max_acceleration_scaling_factor(factor)
                arm.arm_group.set_max_velocity_scaling_factor(factor)
                # X Y 0.306183707634, 0.0385578174025
                tp = arm.get_cartesian_pose()
                tp.position.x = 0.306183707634 + x_[j]
                tp.position.y = 0.0385578174025 + y_[j]
                success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None, wait=True)

                # Q
                # arm.arm_group.set_max_acceleration_scaling_factor(0.1)
                # arm.arm_group.set_max_velocity_scaling_factor(0.1)
                # tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = 0.0, 0.70707811564, 0.0, 0.707135444191
                # success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None, wait=True)
                #####
                # arm.arm_group.set_max_acceleration_scaling_factor(0.1)
                # arm.arm_group.set_max_velocity_scaling_factor(0.1)
                # arm.reach_cartesian_pose(pose=rotate_pose_by_rpy(tp, 0.0, pitch_[i], yaw_[i]), tolerance=tolerance,
                #                          constraints=None)
                ####
                # theta = np.random.uniform(-np.radians(10), np.radians(10))
                # roll, pitch, yaw = tf.transformations.euler_from_quaternion(
                #     (tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w))
                #
                # yaw += yaw_[j]
                # pitch += pitch_[j]
                #


                # quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
                # tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = quaternion
                # success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None)
                # Z
                arm.arm_group.set_max_acceleration_scaling_factor(factor)
                arm.arm_group.set_max_velocity_scaling_factor(factor)
                tp.position.z = -0.05
                success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None, wait=False)
                start_insert_time = rospy.get_time()

                rospy.loginfo('Force , scaling factor ' + str(factor) + ' tolerance: ' + str(tolerance))
                data = []
                data_m = []

                before_height = get_obj_pos()[2]
                first_ft = get_ft_reading()
                first_mt = get_mt_reading()

                ft_, mt_ = first_ft, first_mt

                while abs(ft_[2]) < 1.5 and \
                        arm.get_cartesian_pose().position.z > 0. and \
                        rospy.get_time() - start_insert_time < 15.0 and \
                        np.any(np.abs(first_mt) - np.abs(mt_) < 20) and\
                        abs(get_obj_pos()[2] - before_height) < 0.05 :

                    rospy.logerr(abs(ft_[2]))
                    ft_ = get_ft_reading()
                    mt_ = get_mt_reading()
                    data.append( ft_ - first_ft)
                    data_m.append(mt_ - first_mt)

                rospy.logerr(len(data))
                arm.arm_group.stop()
                rospy.loginfo("stopped")

                rg = 10

                for f in range(rg):
                    suc = move_srv(np.hstack((0, [0,0,0.1]))).success
                for f in range(rg):
                    suc = move_srv(np.hstack((0, [0,0,-0.1]))).success
                for f in range(rg):
                    suc = move_srv(np.hstack((0, [0,0.1,0]))).success
                for f in range(rg):
                    suc = move_srv(np.hstack((0, [0,-0.1,0]))).success
                for f in range(rg):
                    suc = move_srv(np.hstack((0, [0.1,0,0]))).success
                for f in range(rg):
                    suc = move_srv(np.hstack((0, [-0.1,0,0]))).success


                tp = arm.get_cartesian_pose()
                for k in range(4):
                    arm.arm_group.set_max_acceleration_scaling_factor(0.01)
                    arm.arm_group.set_max_velocity_scaling_factor(0.01)
                    arm.reach_cartesian_pose(pose=rotate_pose_by_rpy(tp, 0.0, pitch_[k], yaw_[k]), tolerance=tolerance,
                                             constraints=None)

                data = np.array(data)
                data_m = np.array(data_m)

                combined.append(data[-up_to:])
                combined_mt.append(data_m[-up_to:])

                fig1, axs1 = plt.subplots(6)
                axs1[0].plot(data[-up_to:,0],'o:r')
                axs1[1].plot(data[-up_to:,1],'o:r')
                axs1[2].plot(data[-up_to:,2],'o:r')
                axs1[3].plot(data[-up_to:,3],'o:r')
                axs1[4].plot(data[-up_to:,4],'o:r')
                axs1[5].plot(data[-up_to:,5],'o:r')

                fig2, axs2 = plt.subplots(3)
                axs2[0].plot(data_m[-up_to:,0],'o:g')
                axs2[1].plot(data_m[-up_to:,1],'o:g')
                axs2[2].plot(data_m[-up_to:,2],'o:g')


                path = "/home/osher/catkin_ws/src/insertion_games/src/tests/" + str(j)
                if not os.path.exists(path):
                    os.makedirs(path)

                fig1.savefig(path + '/ft' + str(i) + '.png')
                fig2.savefig(path + '/mt' + str(i) + '.png')
                
                # UP
                arm.arm_group.set_max_acceleration_scaling_factor(0.1)
                arm.arm_group.set_max_velocity_scaling_factor(0.1)
                tp = arm.get_cartesian_pose()
                tp.position.x, tp.position.y, tp.position.z = 0.287988619809, 0.0, 0.05
                tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = 0.5, -0.5, 0.5, 0.5

                success = arm.reach_cartesian_pose(pose=tp, tolerance=tolerance, constraints=None)

                after_press = get_mt_reading()
                delta_a_action = np.where(np.abs(first_mt - after_press) > 10, 0.1, 0.05)
                for _ in range(3):
                    suc = move_srv(np.hstack((0,delta_a_action))).success

                # if np.any(np.abs(before_press - after_press)> 20):
                #     close_srv()

            f = np.dstack(combined)
            mean_comb = np.mean(f, axis=2)
            f2 = np.dstack(combined_mt)
            mean_comb2 = np.mean(f2, axis=2)

            fig1, axs1 = plt.subplots(6)
            axs1[0].plot(mean_comb[-up_to:, 0], 'o:r')
            axs1[1].plot(mean_comb[-up_to:, 1], 'o:r')
            axs1[2].plot(mean_comb[-up_to:, 2], 'o:r')
            axs1[3].plot(mean_comb[-up_to:, 3], 'o:r')
            axs1[4].plot(mean_comb[-up_to:, 4], 'o:r')
            axs1[5].plot(mean_comb[-up_to:, 5], 'o:r')

            fig2, axs2 = plt.subplots(3)
            axs2[0].plot(mean_comb2[-up_to:, 0], 'o:g')
            axs2[1].plot(mean_comb2[-up_to:, 1], 'o:g')
            axs2[2].plot(mean_comb2[-up_to:, 2], 'o:g')

            fig1.suptitle('x: ' + str(x_[j]) + '   ' 'y: ' + str(y_[j]), fontsize=16)
            fig2.suptitle('x: ' + str(x_[j]) + '   ' 'y: ' + str(y_[j]), fontsize=16)

            fig1.savefig(path + '/ft' + 'comb.png')
            fig2.savefig(path + '/mt' + 'comb.png')

    if False:
        i, num_steps = 0, 50
        while i< num_steps and not cond:
            fix_x, fix_y = [np.random.uniform(-0.01, 0.01) for _ in range(2)]
            fix_theta = np.random.uniform(-np.radians(5), np.radians(5))
            tp = arm.get_cartesian_pose()
            tp.position.x += fix_x
            tp.position.y += fix_y
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(
                (tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w))
            yaw += fix_theta
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = quaternion
            cartesian_plan, fraction = arm.plan_cartesian_path(tp)
            success = arm.execute_plan(cartesian_plan)
            cond = abs(get_ft_reading().wrench.force.z) < 2
            print(fix_x, fix_y,fix_theta,get_ft_reading().wrench.force.z)
            i+=1

    if with_openhand:
        if success: rospy.loginfo(success)
        close_srv()




if __name__ == '__main__':
    main()
