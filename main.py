#https://pysource.com
import cv2
from RealsenseCamera import *
from MaskRCNN import *
from Obj import *





# with RealsenseCamera() as rs:
#     mrcnn = MaskRCNN()
#     while True:
#         # Get frame in real time from Realsense camera
#         bgr_frame, depth_image, depth_frame = rs.get_frame_stream()

#         # Get object mask
#         objs = mrcnn.detect_objects_mask(bgr_frame)

#         intr = rs.get_intrinsics()

#         # Draw objects masks and info
#         for i, obj in enumerate(objs):
#             obj.draw_mask(bgr_frame)
#             obj.draw_info(bgr_frame, depth_image, intr, depth_frame)

#         cv2.imshow("depth frame", depth_image)
#         cv2.imshow("Bgr frame", bgr_frame)

#         key = cv2.waitKey(1)
#         if key == 27:
#             break
#     cv2.destroyAllWindows()

rs = RealsenseCamera()
mrcnn = MaskRCNN()
while True:
    # Get frame in real time from Realsense camera
    bgr_frame, depth_image, depth_frame = rs.get_frame_stream()

    # Get object mask
    objs = mrcnn.detect_objects_mask(bgr_frame)

    intr = rs.get_intrinsics()

    # Draw objects masks and info
    for i, obj in enumerate(objs):
        obj.draw_mask(bgr_frame)
        obj.draw_info(bgr_frame, intr, depth_frame)

    cv2.imshow("depth frame", depth_image)
    cv2.imshow("Bgr frame", bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
rs.release()