#!/usr/bin/env python

import rospy
from finger_ros import TactileFingerROSPublisher

if __name__ == "__main__":

    rospy.init_node('tactile_finger_publisher', anonymous=True)

    dev_name = rospy.get_param('~dev_name', 2)
    # when connecting - 4 - 0 - 2
    # dev_names = [4, 2, 0]  # left, right, bottom

    if dev_name == 4:  # left
        fix = (2, 12)
    elif dev_name == 0:  # bottom
        fix = (8, 10)
    elif dev_name == 2:  # right
        fix = (15, 15)
    else:
        assert 'fix this udevrule shit'

    tactile = TactileFingerROSPublisher(dev_name=dev_name, serial='/dev/video', fix=fix)

    tactile.connect()

    tactile.run()