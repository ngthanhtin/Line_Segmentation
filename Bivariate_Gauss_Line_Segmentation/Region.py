
"""
Class representing the line regions
"""
from Line import Line
import numpy as np
import cv2
import math
from numpy.linalg import inv, det


class Region():
    def __init__(self, top=Line(), bottom=Line()):
        """
        region_id: region's id
        region: 2d matrix representing the region
        top: Lines representing region top boundaries
        bottom: Lines representing region bottom boundaries
        height: Region's height
        row_offset: the offset of each col to the original image matrix
        covariance:
        mean: The mean of
        """
        self.top = top
        self.bottom = bottom

        self.region_id = 0
        self.height = 0

        self.region = np.array(()) # used for binary image

        self.row_offset = 0
        self.covariance = np.zeros([2, 2], dtype=np.float32)
        self.mean = np.zeros((1, 2))

    def update_region(self, gray_image, region_id):
        self.region_id = region_id

        if self.top.initial_valley_id == -1:  # none
            min_region_row = 0
            self.row_offset = 0
        else:
            min_region_row = self.top.min_row_pos
            self.row_offset = self.top.min_row_pos

        if self.bottom.initial_valley_id == -1:  # none
            max_region_row = gray_image.shape[0]  # rows
        else:
            max_region_row = self.bottom.max_row_pos

        start = int(min(min_region_row, max_region_row))
        end = int(max(min_region_row, max_region_row))

        self.region = np.ones((end - start, gray_image.shape[1]), dtype=np.uint8) * 255

        # Fill region.
        for c in range(gray_image.shape[1]):
            if len(self.top.valley_ids) == 0:
                start = 0
            else:
                if len(self.top.points) != 0:
                    start = self.top.points[c][0]
            if len(self.bottom.valley_ids) == 0:
                end = gray_image.shape[0] - 1
            else:
                if len(self.bottom.points) != 0:
                    end = self.bottom.points[c][0]

            # Calculate region height
            if end > start:
                self.height = max(self.height, end - start)

            for i in range(int(start), int(end)):
                self.region[i - int(min_region_row)][c] = gray_image[i][c]

        self.calculate_mean()
        self.calculate_covariance()

        return True
        # return cv2.countNonZero(self.region) == (self.region.shape[0] * self.region.shape[1])

    def calculate_mean(self):
        self.mean[0][0] = 0.0
        self.mean[0][1] = 0.0
        n = 0
        for i in range(self.region.shape[0]):
            for j in range(self.region.shape[1]):
                # if white pixel continue.
                if self.region[i][j] == 255.0:
                    continue
                if n == 0:
                    n = n + 1
                    self.mean[0][0] = i + self.row_offset
                    self.mean[0][1] = j
                else:
                    vec = np.zeros((1,2))
                    vec[0][0] = i + self.row_offset
                    vec[0][1] = j
                    self.mean = ((n - 1.0) / n) * self.mean + (1.0 / n) * vec
                    n = n + 1
        # print(self.mean)

    def calculate_covariance(self):
        n = 0  # Total number of considered points (pixels) so far
        self.covariance = np.zeros([2, 2], dtype=np.float32)
        sum_i_squared = 0
        sum_j_squared = 0
        sum_i_j = 0

        for i in range(self.region.shape[0]):
            for j in range(self.region.shape[1]):
                # if white pixel continue
                if int(self.region[i][j]) == 255:
                    continue

                new_i = i + self.row_offset - self.mean[0][0]
                new_j = j - self.mean[0][1]

                sum_i_squared += new_i * new_i
                sum_i_j += new_i * new_j
                sum_j_squared += new_j * new_j
                n += 1

        if n:
            self.covariance[0][0] = sum_i_squared / n
            self.covariance[0][1] = sum_i_j / n
            self.covariance[1][0] = sum_i_j / n
            self.covariance[1][1] = sum_j_squared / n

    def bi_variate_gaussian_density(self, point):
        point[0][0] -= self.mean[0][0]
        point[0][1] -= self.mean[0][1]
        # print(point)
        point_transpose = np.transpose(point)
        ret = ((point * inv(self.covariance) * point_transpose))
        ret *= np.sqrt(det(2 * math.pi * self.covariance))
        # print("Ret[0][0]: {}".format(ret[0][0]))
        return ret[0][0]
