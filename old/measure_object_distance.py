#https://pysource.com
import cv2
from realsense_camera import *
from mask_rcnn import *
import math


def get_xyz(intr,depth_frame,center):
	theta = 0
	dist = depth_frame.get_distance(int(center[0]), int(center[1]))*1000 #convert to mm

	#calculate real world coordinates
	Xtemp = dist*(center[0] -intr.ppx)/intr.fx
	Ytemp = dist*(center[1] -intr.ppy)/intr.fy
	Ztemp = dist

	Xtarget = Xtemp - 35 #35 is RGB camera module offset from the center of the realsense
	Ytarget = -(Ztemp*math.sin(theta) + Ytemp*math.cos(theta))
	Ztarget = Ztemp*math.cos(theta) + Ytemp*math.sin(theta)

	return Xtarget, Ytarget, Ztarget



# Load Realsense camera
rs = RealsenseCamera()
mrcnn = MaskRCNN()

while True:
	# Get frame in real time from Realsense camera
	ret, bgr_frame, depth_image, depth_frame = rs.get_frame_stream()

	# Get object mask
	boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

	intr = rs.get_intrinsics()

	for i,center in enumerate(centers):
		print(f"{i+1}: {get_xyz(intr, depth_frame, center)}")
		
	# Draw object mask
	bgr_frame = mrcnn.draw_object_mask(bgr_frame)

	# Show depth info of the objects
	mrcnn.draw_object_info(bgr_frame, depth_image,intr,depth_frame)


	cv2.imshow("depth frame", depth_image)
	cv2.imshow("Bgr frame", bgr_frame)

	key = cv2.waitKey(1)
	if key == 27:
		break

rs.release()
cv2.destroyAllWindows()
