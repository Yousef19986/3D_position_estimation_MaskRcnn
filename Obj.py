import scipy
import numpy as np
import cv2


class Obj:
    """
    Class: Obj
    Attributes:
        class_name:     string
        box:            tuple (x,y,x2,y2)
        center:         tuple (cx,cy)
        contour:        list  (numpy.array)
        color:          tuple (B,G,R)

    Methods:
        get_box():                      returns box
        get_center():                   returns center
        get_contour():                  returns contour
        get_color():                    returns color
        get_class_name():               returns class_name

        draw_mask(frame):               draws mask on frame
        draw_info(frame, intr, depth):  draws info on frame
        get_coordinates(intr, depth):   returns real world coordinates
    """

    def __init__(self, class_name, box, center, contour, color=None):
        """
        Constructor
        :param class_name:     string
        :param box:            tuple (x,y,x2,y2)
        :param center:         tuple (cx,cy)
        :param contour:        list  (numpy.array)
        :param color:          tuple (B,G,R)
        """
        self.class_name = class_name
        self.box = box
        self.center = center
        self.contour = contour

        if color is None:
            # generate random color (BGR)
            self.color = np.random.randint(0, 256, (3))
        else:
            self.color = color

        # convert color np.array to tuple
        self.color = tuple(self.color)

        print(self.color)

    def get_box(self):
        return self.box

    def get_center(self):
        return self.center

    def get_contour(self):
        return self.contour

    def get_color(self):
        return self.color

    def get_class_name(self):
        return self.class_name

    def draw_mask(self, frame):
        """
        Method: draw_mask
        description: 
            draws mask on frame
            :param frame: numpy.array
            :return:      updated frame
        """

        x, y, x2, y2 = self.box
        roi = frame[y: y2, x: x2]
        roi_copy = np.zeros_like(roi)

        for cnt in self.contour:
            cv2.drawContours(roi, cnt, -1, self.color, 3)
            cv2.fillPoly(roi_copy, [cnt], self.color)
            roi = cv2.addWeighted(roi, 1, roi_copy, 0.5, 0.0)
            frame[y: y2, x: x2] = roi

        return frame

    def get_coordinates(self, intr, depth):
        """
        Method: get_coordinates
        description:
            returns real world coordinates
            :param intr:     realsense intrinsics
            :param depth:    original depth frame
            :return:         real world coordinates (x,y,z)
        """

        theta = 0
        dist = depth.get_distance(int(self.center[0]), int(
            self.center[1]))*1000  # convert to mm

        #calculate real world coordinates
        Xtemp = dist*(self.center[0] - intr.ppx)/intr.fx
        Ytemp = dist*(self.center[1] - intr.ppy)/intr.fy
        Ztemp = dist

        Xtarget = Xtemp - 35  # 35 is RGB camera module offset from the center of the realsense
        Ytarget = -(Ztemp*scipy.sin(theta) + Ytemp*scipy.cos(theta))
        Ztarget = Ztemp*scipy.cos(theta) + Ytemp*scipy.sin(theta)

        return Xtarget, Ytarget, Ztarget

    def draw_info(self, frame, intr, depth):
        """
        Method: draw_info
        description:
            draws info on frame (class name, center, real world coordinates)
            :param frame:    numpy.array
            :param intr:     realsense intrinsics
            :param depth:    original depth frame
            :return:         updated frame
        """

        x, y, x2, y2 = self.box
        cx, cy = self.center
        cv2.line(frame, (cx, y), (cx, y2), self.color, 1)
        cv2.line(frame, (x, cy), (x2, cy), self.color, 1)

        cv2.rectangle(frame, (x, y), (x + 250, y + 70), self.color, -1)
        cv2.putText(frame, self.class_name.capitalize(),
                    (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{self.get_coordinates(intr,depth,(cx,cy))}",
                    (x + 5, y + 60), 0, 0.5, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x2, y2), self.color, 1)
        cv2.circle(frame, (cx, cy), 1, (255, 255, 255), 2)

        return frame
