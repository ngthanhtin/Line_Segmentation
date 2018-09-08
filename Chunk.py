
from Peak import Peak
from Valley import Valley
import numpy as np
import cv2
import math

"""
Class Chunk represents the vertical segment cut
There are 20 CHUNK, because each every chunk is 5% of a iamge
"""
class Chunk():
    def __init__(self, index = 0, start_col = 0, width = 0, img = np.array(())):
        """
        index: index of the chunk
        start_col: the start column positition
        width: the width of the chunk
        img: gray iamge
        histogram: the value of the y histogram projection profile
        peaks: found peaks in this chunk
        valleys: found valleys in this chunk
        avg_height: average line height in this chunk
        avg_white_height: average space height in this chunk
        lines_count: the estimated number of lines in this chunk
        """
        self.index = index
        self.start_col = start_col
        self.width = width
        self.thresh_img = img.copy()

        self.histogram = []  # length is the number of rows in an image
        for i in range(self.thresh_img.shape[0]):  #rows
            self.histogram.append(0)
        self.peaks = [] # Peak type
        self.valleys = [] #Valley type
        self.avg_height = 0
        self.avg_white_height = 0
        self.lines_count = 0
    
    def calculate_histogram(self):
        # get the smoothed profile by applying a median filter of size 5
        cv2.medianBlur(self.thresh_img, 5, self.thresh_img)

        current_height = 0
        current_white_count = 0
        white_lines_count = 0

        white_spaces = []
        
        rows, cols = self.thresh_img.shape[:2]

        for i in range(rows):
            black_count = 0
            for j in range(cols):
                if self.thresh_img[i][j] == 0:
                    black_count += 1
                    self.histogram[i] += 1
            if black_count:
                current_height += 1
                if current_white_count:
                    white_spaces.append(current_white_count)
                current_white_count = 0
            else:
                current_white_count += 1
                if current_height:
                    self.lines_count += 1
                    self.avg_height += current_height
                current_height = 0
        #calculate the white spaces average height
        white_spaces.sort()  # sort ascending
        for i in range(len(white_spaces)):
            if white_spaces[i] > 4 * self.avg_height:
                break
            self.avg_white_height += white_spaces[i]
            white_lines_count+=1
        
        if white_lines_count:
            self.avg_white_height /= white_lines_count
        #calculate the average line height
        if self.lines_count:
            self.avg_height /= self.lines_count
        
        # 30 is hyper-parameter
        self.avg_height = max(30, int(self.avg_height + self.avg_height / 2.0))


    def find_peaks_valleys(self, map_valley = {}):
        self.calculate_histogram()
        #detect peaks
        len_histogram = len(self.histogram)

        for i in range(1, len_histogram - 1):
            left_val = self.histogram[i - 1]
            centre_val = self.histogram[i]
            right_val = self.histogram[i + 1]
            #peak detection
            if centre_val >= left_val and centre_val >= right_val:
                # Try to get the largest peak in same region.
                if len(self.peaks) != 0 and i - self.peaks[-1].position <= self.avg_height // 2 and centre_val >= self.peaks[-1].value:
                    self.peaks[-1].position = i
                    self.peaks[-1].value = centre_val
                elif len(self.peaks) > 0 and i - self.peaks[-1].position <= self.avg_height // 2 and centre_val < self.peaks[-1].value:
                    abc = 0
                else:
                    self.peaks.append(Peak(position=i, value=centre_val))
        #
        peaks_average_values = 0
        new_peaks = []  # Peak type
        for p in self.peaks:
            peaks_average_values += p.value
        peaks_average_values //= max(1, int(len(self.peaks)))

        for p in self.peaks:
            if p.value >= peaks_average_values / 4:
                new_peaks.append(p)
        
        self.lines_count = int(len(new_peaks))

        self.peaks = new_peaks
        #sort peaks by max value and remove the outliers (the ones with less foreground pixels)
        self.peaks.sort(key=Peak().get_value)
        #resize self.peaks
        if self.lines_count + 1 <= len(self.peaks):
            self.peaks = self.peaks[:self.lines_count + 1]
        else:
            self.peaks = self.peaks[:len(self.peaks)]
        self.peaks.sort(key=Peak().get_row_position)

        #search for valleys between 2 peaks
        for i in range(1, len(self.peaks)):
            min_pos = (self.peaks[i - 1].position + self.peaks[i].position) / 2
            min_value = self.histogram[int(min_pos)]
            
            start = self.peaks[i - 1].position + self.avg_height / 2
            end = 0
            if i == len(self.peaks):
                end = self.thresh_img.shape[0]  #rows
            else:
                end = self.peaks[i].position - self.avg_height - 30

            for j in range(int(start), int(end)):
                valley_black_count = 0
                for l in range(self.thresh_img.shape[1]):  #cols
                    if self.thresh_img[j][l] == 0:
                        valley_black_count += 1
                
                if i == len(self.peaks) and valley_black_count <= min_value:
                    min_value = valley_black_count
                    min_pos = j
                    if min_value == 0:
                        min_pos = min(self.thresh_img.shape[0] - 10, min_pos + self.avg_height)
                        j = self.thresh_img.shape[0]
                elif min_value != 0 and valley_black_count <= min_value:
                    min_value = valley_black_count
                    min_pos = j
            
            new_valley = Valley(chunk_index=self.index, position=min_pos)
            self.valleys.append(new_valley)
            
            # map valley
            map_valley[new_valley.valley_id] = new_valley
        return int(math.ceil(self.avg_height))


                
