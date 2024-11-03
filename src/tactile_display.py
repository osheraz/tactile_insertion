#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt

def circle_mask(size=(640, 480), border=0, fix=(0,0)):
    """
        used to filter center circular area of a given image,
        corresponding to the AllSight surface area
    """
    m = np.zeros((size[1], size[0]))

    m_center = (size[0] // 2 - fix[0], size[1] // 2 - fix[1])
    m_radius = min(size[0], size[1]) // 2 - border - max(abs(fix[0]), abs(fix[1]))
    m = cv2.circle(m, m_center, m_radius, 255, -1)
    m /= 255
    m = m.astype(np.float32)
    mask = np.stack([m, m, m], axis=2)

    return mask


def plot_histogram_comparison(rgb_image, gray_image, axes):
    """Update histograms of both RGB and grayscale images."""
    colors = ('r', 'g', 'b')  # Red, Green, Blue channels

    # Clear previous histograms
    for ax in axes:
        ax.clear()

    # Plot RGB channel histograms
    for i, color in enumerate(colors):
        channel = rgb_image[:, :, i].flatten()  # Flatten each channel
        axes[0].hist(channel, bins=50, range=(0, 1), color=color, alpha=0.5, label=f'{color.upper()} Channel')

    axes[0].set_title("RGB Channel Histograms (Normalized)")
    axes[0].set_xlabel("Pixel Intensity (Normalized)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(loc='upper right')

    # Plot grayscale histogram
    gray_flat = gray_image.flatten()  # Flatten grayscale image
    axes[1].hist(gray_flat, bins=50, range=(0, 1), color='black', alpha=0.7)

    axes[1].set_title("Grayscale Histogram (Normalized)")
    axes[1].set_xlabel("Pixel Intensity (Normalized)")
    # axes[1].set_xlim([0.4,0.7])
    axes[1].set_ylabel("Frequency")

    # Redraw the updated plots
    plt.draw()
    plt.pause(0.0000000000001)  # Pause to allow the plot to update

class ImageMerger:
    def __init__(self):
        rospy.init_node('image_merger')

        self._cv_bridge = CvBridge()
        self.images = [None, None, None]  # Placeholder for the 3 images
        self.reference_images = [None, None, None]  # Placeholder for the reference images
        self.diff_images = [None, None, None]
        self.got_ref = False
        # dev_names = [4, 2, 0]  # left, right, bottom
        # new_names = [0, 4, 2]  # left, right, bottom
        # Subscribers for the three image topics
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        rospy.Subscriber("allsight0/usb_cam/image_raw", Image, self.image_callback_bottom, callback_args=2,
                         queue_size=1)  # Left
        rospy.Subscriber("allsight2/usb_cam/image_raw", Image, self.image_callback_right, callback_args=1,
                         queue_size=1)  # Bottom
        rospy.Subscriber("allsight4/usb_cam/image_raw", Image, self.image_callback_left, callback_args=0,
                         queue_size=1)  # Right

        ros_rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            try:
                self.update_and_publish()
                ros_rate.sleep()

            except RuntimeError as e:
                rospy.logerr(e)

    def _subtract_bg(self, img1, img2, offset=0.6):
        """
        Subtract background image from foreground image and normalize the result.

        :param img1: Foreground image
        :param img2: Background image
        :param offset: Normalization offset
        :return: Normalized difference image
        """
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0
        # cv2.cvtColor(diff.astype('float32'), cv2.COLOR_BGR2GRAY)

        # diff[diff < -0.00] = 0
        # print(diff.min())
        # print(diff.max())
        diff += offset
        # print(f"Min pixel value: {np.min(diff)}")
        # print(f"Max pixel value: {np.max(diff)}")
        # print(f"Mean pixel value: {np.mean(diff)}")
        # print(f"Standard deviation: {np.std(diff)}")
        return diff

    def image_callback(self, data, idx):
        """
        Callback function to handle incoming image data.

        :param data: ROS Image message
        :param idx: Index of the image source
        """
        try:
            cv_image = self._cv_bridge.imgmsg_to_cv2(data, "bgr8")
            self.images[idx] = cv_image
            if self.got_ref:
                self._update_diff_images()
                self._try_merge_images()
            else:
                if all(image is not None for image in self.images):
                    self._initialize_reference_images()

        except CvBridgeError as e:
            rospy.logerr("Error converting ROS Image message to CV image: ", e)

    def image_callback_bottom(self, data, idx):
        """
        Callback function to handle incoming image data.

        :param data: ROS Image message
        :param idx: Index of the image source
        """
        try:
            cv_image = self._cv_bridge.imgmsg_to_cv2(data, "bgr8")
            self.images[idx] = cv_image
        except CvBridgeError as e:
            rospy.logerr("Error converting ROS Image message to CV image: ", e)

    def image_callback_left(self, data, idx):
        """
        Callback function to handle incoming image data.

        :param data: ROS Image message
        :param idx: Index of the image source
        """
        try:
            cv_image = self._cv_bridge.imgmsg_to_cv2(data, "bgr8")
            self.images[idx] = cv_image
        except CvBridgeError as e:
            rospy.logerr("Error converting ROS Image message to CV image: ", e)

    def image_callback_right(self, data, idx):
        """
        Callback function to handle incoming image data.

        :param data: ROS Image message
        :param idx: Index of the image source
        """
        try:
            cv_image = self._cv_bridge.imgmsg_to_cv2(data, "bgr8")
            self.images[idx] = cv_image
        except CvBridgeError as e:
            rospy.logerr("Error converting ROS Image message to CV image: ", e)

    def update_and_publish(self):

        if self.got_ref:
            self._update_diff_images()
            self._try_merge_images()
        else:
            if all(image is not None for image in self.images):
                self._initialize_reference_images()

    def _initialize_reference_images(self):
        """
        Initialize reference images and compute mask for the merging process.
        """
        self.reference_images = [im for im in self.images]
        self.min_height = min(image.shape[0] for image in self.images)
        self.min_width = min(image.shape[1] for image in self.images)
        self.mask = circle_mask((224, 224))
        self.got_ref = True

    def _update_diff_images(self):
        """
        Update the difference images by subtracting the background images.
        """
        self.diff_images = [self._subtract_bg(fg, bg) for (fg, bg) in zip(self.images, self.reference_images)]

    def _try_merge_images(self):
        """
        Try to merge the difference images and display the result.
        """
        if all(image is not None for image in self.diff_images):
            # resized_images = [cv2.resize(image, (224, 244)) * self.mask for image in
            #                   self.diff_images]
            resized_images = [
                cv2.resize(image, (224, 224)) * self.mask
                for image in self.diff_images
            ]
            # resized_images = [i.astype(np.uint8) for i in resized_images]

            merged_image = np.hstack(resized_images)
            merged_image = merged_image[: merged_image.shape[0]//2, :, :]
            cv2.imshow("Merged Image", merged_image)
            cv2.waitKey(1)
        else:
            rospy.logwarn("Failed to merge images: Not all difference images are available")


    def _try_merge_images3(self):
        """Merge the difference images and display them."""
        if all(image is not None for image in self.diff_images):
            resized_images = [
                cv2.resize(image, (224, 224)) * self.mask
                for image in self.diff_images
            ]
            merged_image = np.hstack(resized_images)[: 224 // 2, :224, :]

            # Display merged image with OpenCV
            cv2.imshow("Merged Image", merged_image)

            # Convert merged image to grayscale
            gray_image = cv2.cvtColor((merged_image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0

            # Plot histograms using Matplotlib
            plot_histogram_comparison(merged_image, gray_image, self.axes)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("User exit")
                plt.close()
                cv2.destroyAllWindows()
        else:
            rospy.logwarn("Failed to merge images: Not all difference images are available")


    def try_merge_images2(self):

        if all(image is not None for image in self.diff_images):
            # Find the minimum height and width among all images
            # Resize all images to the smallest dimensions found
            resized_images = [cv2.resize(image, (self.min_width, self.min_height)) * self.mask for image in self.diff_images]
            # Merge resized images horizontally
            idx = 1
            merged_image = resized_images[idx]

            # Display the merged image
            cv2.imshow("diff image", merged_image)
            cv2.imshow("raw image", self.images[idx])

            cv2.waitKey(1)
        else:
            print('failed')

if __name__ == '__main__':
    try:
        image_merger = ImageMerger()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
