# https://pysource.com/instance-segmentation-mask-rcnn-with-python-and-opencv
import cv2
import numpy as np
from Obj import Obj


class MaskRCNN:
    """
    Class: MaskRCNN 
    Methods:
        detect_objects_mask(bgr_frame): returns list of objects
    Attributes:
        net:                            Mask RCNN
        detection_threshold:            Confidence threshold
        mask_threshold:                 Mask threshold
        colors:                         Random colors
        classes:                        Classes_names
        objects:                        List of detected objects in frame
    """

    def __init__(self, detection_threshold=0.7, mask_threshold=0.3):
        """
        Constructor
        description:
            Initialize Mask RCNN
        :param detection_threshold: Confidence threshold
        :param mask_threshold:      Mask threshold
        """

        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                                 "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Conf threshold
        self.detection_threshold = detection_threshold
        self.mask_threshold = mask_threshold

        # Generate random colors
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        self.classes = []
        with open("dnn/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.objects = []

    def detect_objects_mask(self, bgr_frame):
        """
        Method: detect_objects_mask
        description:
            Detect objects in frame and return list of objects
        :param bgr_frame:   BGR frame
        :return:            List of objects
        """
        blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(
            ["detection_out_final", "detection_masks"])

        # Detect objects
        frame_height, frame_width, _ = bgr_frame.shape
        detection_count = boxes.shape[2]

        for i in range(detection_count):
            box = boxes[0, 0, i]
            name = box[1]
            score = box[2]
            if score < self.detection_threshold:
                continue

            # Get box Coordinates
            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            obj_box = [x, y, x2, y2]

            # Get Object Center
            cx = (x + x2) // 2
            cy = (y + y2) // 2
            center = (cx, cy)

            # Contours
            # Get mask coordinates
            # Get the mask
            mask = masks[i, int(name)]
            roi_height, roi_width = y2 - y, x2 - x
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(
                mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntr = contours

            self.objects.append(Obj(name, obj_box, center, cntr, self.color[i]))

        return self.objects
