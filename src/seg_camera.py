#!/usr/bin/env python

import numpy as np
import rospy

from sensor_msgs.msg import Image
from deploy_utils import msg_to_pil, image_msg_to_numpy
import numpy
import cv2
from FastSAM.live_sam import FastSAM, FastSAMPrompt
import time
from cv_bridge import CvBridge


def masks_to_bool(masks):
    if type(masks) == numpy.ndarray:
        return masks.astype(bool)
    return masks.cpu().numpy().astype(bool)


class SegCameraSubscriber:

    def __init__(self, with_socket=True, topic='/zedm/zed_node/rgb/image_rect_color', device='cuda:0'):
        """
        """
        self.mask_pub = rospy.Publisher('/seg_mask_image', Image, queue_size=10)
        self.raw_pub = rospy.Publisher('/seg_raw_image', Image, queue_size=10)

        self.last_frame = None
        self.raw_frame = None
        self.w = 320
        self.h = 180
        self.init_success = True
        self.device = device
        self.img_size = 1024
        self.conf = 0.4
        self.iou = 0.9
        self.bridge = CvBridge()
        # Default configurations
        self.table_dims = {"x_min": 10, "y_min": 10, "x_max": 280, "y_max": 250}
        self.socket_rough_pos = {"x_min": 120, "y_min": 110, "x_max": 280, "y_max": 180}
        self.max_dims = {"width": 70, "height": 70}
        self.socket_max_dims = {"width": 60, "height": 60}
        self.socket_min_dims = {"width": 0, "height": 0}

        self.min_dims = {"width": 10, "height": 15}
        self.with_socket = with_socket

        self.got_socket_mask = False
        self.points_to_exclude = [(0, 0)]  # [(147, 156), (147, 156)]
        self.socket_id = 3
        self.plug_id = 2
        self._topic_name = rospy.get_param('~topic_name', '{}'.format(topic))
        rospy.loginfo("(topic_name) Subscribing to Images to topic  %s", self._topic_name)
        self.model = FastSAM('/home/roblab20/osher3_workspace/src/isaacgym/python/'
                             'IsaacGymInsertion/isaacgyminsertion/outputs/weights/FastSAM-x.pt')
        self._image_subscriber = rospy.Subscriber(self._topic_name, Image, self.image_callback, queue_size=1)
        self._check_camera_ready()

    def _check_camera_ready(self):

        self.last_frame = None
        rospy.logdebug(
            "Waiting for '{}' to be READY...".format(self._topic_name))
        while self.last_frame is None and not rospy.is_shutdown():
            try:
                self.last_frame = rospy.wait_for_message(
                    '{}'.format(self._topic_name), Image, timeout=5.0)
                rospy.logdebug(
                    "Current '{}' READY=>".format(self._topic_name))
                self.start_time = rospy.get_time()
                # init model overrides
                _ = self.model.predict(source=self.last_frame,
                                       device=self.device,
                                       retina_masks=True,
                                       imgsz=640,
                                       conf=0.4,
                                       iou=0.9, )
            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(self._topic_name))
        return self.last_frame

    def is_box_within_rect(self, box, rect, exclude_points):
        x_min, y_min, x_max, y_max = box

        includes_exclude_points = any(
            x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
            for point in exclude_points
        )

        within_rect = (
                x_min >= rect["x_min"] and y_min >= rect["y_min"] and
                x_max <= rect["x_max"] and y_max <= rect["y_max"]
        )

        return within_rect and not includes_exclude_points

    def is_box_within_rect_and_dim(self, box, rect, max_dim, min_dim, exclude_points):
        x_min, y_min, x_max, y_max = box
        box_width = x_max - x_min
        box_height = y_max - y_min

        within_rect_and_dim = (
                x_min >= rect["x_min"] and y_min >= rect["y_min"] and
                x_max <= rect["x_max"] and y_max <= rect["y_max"] and
                min_dim["width"] < box_width < max_dim["width"] and
                min_dim["height"] < box_height < max_dim["height"]
        )

        includes_exclude_points = any(
            x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
            for point in exclude_points
        )

        return within_rect_and_dim and not includes_exclude_points

    def find_smallest_and_largest_boxes(self, socket_boxes):
        if not socket_boxes:
            return None, None  # Return None if the list is empty

        def box_area(box):
            x_min, y_min, x_max, y_max = box
            return (x_max - x_min) * (y_max - y_min)

        smallest_box = min(socket_boxes, key=box_area)
        largest_box = max(socket_boxes, key=box_area)

        return smallest_box, largest_box

    def image_callback(self, msg):
        try:
            self.raw_frame = image_msg_to_numpy(msg)
            # self.process_frame(self.raw_frame)
        except Exception as e:
            print(e)
            return

    def process_frame(self, frame):

        start_time = time.time()
        # Resize the frame
        frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # Perform model prediction
        results = self.model.predict(
            source=frame,
            device=self.device,
            retina_masks=True,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
        )

        seg, socket, all_boxes = [], [], []

        for box in results[0].boxes:
            box = box.xyxy.cpu().numpy()[0]
            if self.is_box_within_rect(box, self.table_dims, self.points_to_exclude):
                all_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            if self.is_box_within_rect_and_dim(box,
                                               self.table_dims,
                                               self.max_dims,
                                               self.min_dims,
                                               self.points_to_exclude):
                seg.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

            if self.is_box_within_rect_and_dim(box,
                                               self.socket_rough_pos,
                                               self.socket_max_dims,
                                               self.socket_min_dims,
                                               self.points_to_exclude):
                socket.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        #
        # for b in all_boxes:
        #     cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

        if not self.got_socket_mask:

            prompt_process = FastSAMPrompt(frame, results, device=self.device)
            hole_box, socket_box = self.find_smallest_and_largest_boxes(socket)
            ann = prompt_process.box_prompt(bbox=socket_box)[0]
            ann_hole = prompt_process.box_prompt(bbox=hole_box)[0]
            mask_socket = masks_to_bool(ann)
            mask_hole = masks_to_bool(ann_hole)
            self.socket_mask = (mask_socket & ~mask_hole).astype(int)
            self.socket_mask *= self.socket_id
            self.original_socket_mask = self.socket_mask.copy()
            self.img_size = 896
            self.conf = 0.6
            self.iou = 0.6
            self.got_socket_mask = True
            self.init_success = True
            self.min_dims = {"width": 15, "height": 30}
            self.max_dims = {"width": 40, "height": 70}

        try:
            prompt_process = FastSAMPrompt(frame, results, device=self.device)
            if len(seg) > 1:
                smallest, biggest = self.find_smallest_and_largest_boxes(seg)
            else:
                smallest = seg[0]

            ann = prompt_process.box_prompt(bbox=smallest)[0]
            self.plug_mask = (masks_to_bool(ann)).astype(int) * self.plug_id
            self.plug_mask = self.shrink_mask(self.plug_mask.astype(float)).astype(int)
            self.socket_mask = np.where(self.plug_mask > 0, 0, self.original_socket_mask)
            self.last_frame = self.plug_mask | self.socket_mask if self.with_socket else self.plug_mask

            self.publish_images(frame)

        except Exception as e:
            print(e)

        return self.last_frame


    def publish_images(self, frame):

        if self.with_socket:
            mask = ((self.last_frame == self.plug_id) | (self.last_frame == self.socket_id)).astype(float)
        else:
            mask = (self.last_frame == self.plug_id).astype(float)

        self.mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        seg_show = (self.mask_3d * frame).astype(np.uint8)

        # Convert images to ROS messages
        mask_msg = self.bridge.cv2_to_imgmsg(seg_show, encoding='bgr8')
        raw_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # Publish the images
        self.mask_pub.publish(mask_msg)
        self.raw_pub.publish(raw_msg)


    def shrink_mask(self, mask, shrink_percentage=10):
        """
        Shrink the object in the mask by a certain percentage of its area.

        :param mask: Binary mask image (object = 255, background = 0)
        :param shrink_percentage: Percentage to shrink the object
        :return: Shrunk mask
        """
        # Calculate the original area of the object
        original_area = np.sum(mask > 0)

        # Calculate the target area after shrinkage
        target_area = original_area * (1 - shrink_percentage / 100.0)

        # Create a structuring element (kernel) for erosion
        kernel = np.ones((3, 3), np.uint8)  # Small 3x3 kernel

        # Iteratively erode the mask until the target area is reached
        shrunk_mask = mask.copy()
        while np.sum(shrunk_mask > 0) > target_area:
            shrunk_mask = cv2.erode(shrunk_mask, kernel, iterations=1)

        return shrunk_mask

    def get_raw_frame(self):
        # rospy.wait_for_message('/zedm/zed_node/rgb/image_rect_color')
        return self.raw_frame if not isinstance(self.raw_frame, Image) else None

    def get_frame(self):
        # rospy.wait_for_message('/zedm/zed_node/rgb/image_rect_color')
        return self.last_frame if not isinstance(self.last_frame, Image) else None


if __name__ == "__main__":

    rospy.init_node('Seg')
    seg_cam = SegCameraSubscriber()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        # seg_cam.process_frame(seg_cam.get_raw_frame())
        rate.sleep()