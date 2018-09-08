import cv2
import numpy as np
from Chunk import Chunk
from Line import Line
from Region import Region
import sys

# every chunk has a 5% of the width
CHUNK_NUMBER = 20
CHUNK_TOBE_PROCESSED = 5

def union(a,b):
    """
    :param a: rect
    :param b: rect
    :return: union area 
    """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def intersection(a,b):
    """
    :param a:  rect
    :param b:  rect
    :return: intersection area 
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # does not intersect
    return (x, y, w, h)

def mergeRects(rects):
    """
     rects = [(x,y,w,h),(x,y,w,h),...]
     return merged rects
    """
    merged_rects = []
    for i in range(len(rects)):
        is_repeated = False
        for j in range(i+1, len(rects)):
            rect_tmp = intersection(rects[i], rects[j])
            # Check for intersection / union
            if rect_tmp == ():
                continue
            rect_tmp_area = rect_tmp[2] * rect_tmp[3]
            rect_i_area = rects[i][2] * rects[i][3]
            rect_j_area = rects[j][2] * rects[j][3]

            if ((rect_tmp_area == rect_i_area) or (rect_tmp_area == rect_j_area)):
                is_repeated = True
                rect_tmp = union(rects[i], rects[j]) # Merging
                merged_rect = rect_tmp
                # Push in merged rectangle after checking all the inner loop.
                if j == len(rects) - 2:
                    merged_rects.append(merged_rect)
                # Update the current vector.
                rects[j] = merged_rect

        if is_repeated == False:
            merged_rects.append(rects[i])

    return merged_rects

class LineSegmentation:
    def __init__(self, img=np.array((0, 0)), output_path=""):
        """
        img: image loaded
        output_path:
        chunks: The image chunks list
        lines_region: All the regions the lines found
        avg_line_height: the average height of lines in the image
        """
        self.output_path = output_path
        self.img = img
        self.chunks = []
        self.chunk_width = 0
        self.map_valley = {}  # {int, Valley}
        self.predicted_line_height = 0

        self.initial_lines = []  # Line type
        self.lines_region = []  # region type

        self.avg_line_height = 0

        self.not_primes_list = [0] * 100007
        self.primes = []
        # sieve
        self.sieve()

    def sieve(self):
        self.not_primes_list[0] = 1
        self.not_primes_list[1] = 1
        for i in range(2, 100000):
            if self.not_primes_list[i] == 1:
                continue

            self.primes.append(i)
            for j in range(i * 2, 100000, i):
                self.not_primes_list[j] = 1

    def addPrimesToList(self, n, probPrimes):
        for i in range(len(self.primes)):
            while (n % self.primes[i]):
                n /= self.primes[i]
                probPrimes[i] += 1

    def pre_process_img(self):
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # blur image
        smoothed_img = cv2.blur(self.gray_img, (3, 3), anchor=(-1, -1))

        _, self.thresh = cv2.threshold(smoothed_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # cv2.imwrite(self.output_path + "/Binary_image.jpg", self.thresh)

    def find_contours(self):
        thresh_clone = self.thresh.copy()
        contours = 0
        ret, contours, hierachy = cv2.findContours(image=thresh_clone, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE, offset=(0,0))

        bounding_rect = []
        for c in contours:
            cv2.approxPolyDP(c, 1, True, c)  # apply approximation to polygons
            bounding_rect.append(cv2.boundingRect(c))
        del (bounding_rect[-1])  # coz the last one is a contour containing whole image
        img_clone = self.img.copy()
        # check intersection among rects
        merged_rect = mergeRects(bounding_rect)

        # draw rects on image
        for r in merged_rect:
            (x, y, w, h) = r
            cv2.rectangle(img_clone, (x, y), (x + w, y + h), (0, 0, 255), 2, 8, 0)
        # cv2.imwrite(self.output_path + "/contours.jpg", img_clone)
        self.contours = merged_rect  # save contours AS rectangles
        # # self.contours = []
        # # for rect in merged_rect:
        # #     (x,y,w,h) = rect
        # #     new_contour = []
        # #     for i in range(x, x + w):
        # #         new_contour.append([i, y])
        # #     for i in range(y+1, y + h - 1):
        # #         new_contour.append([x, i])
        # #         new_contour.append([x + w - 1, i])
        # #     for i in range(x, x + w):
        # #         new_contour.append([i, y+h - 1])
        # #     self.contours.append(new_contour)
        # print(self.contours)




    def generate_chunks(self):
        rows, cols = self.thresh.shape[:2]
        width = cols
        self.chunk_width = int(width / CHUNK_NUMBER)

        start_pixel = 0
        for i_chunk in range(CHUNK_NUMBER):
            c = Chunk(index=i_chunk, start_col=start_pixel, width=self.chunk_width,
                      img=self.thresh[0:rows, start_pixel:start_pixel + self.chunk_width].copy())
            self.chunks.append(c)

            # cv2.imwrite(self.output_path + "/Chunk" + str(i_chunk) + ".jpg", self.chunks[-1].gray_img)

            start_pixel += self.chunk_width

    def connect_valleys(self, i, current_valley, line, valleys_min_abs_dist):
        """
        i: the chunk's index
        current_valley
        line
        valleys_min_abs_dist:
        """
        if (i <= 0 or len(self.chunks[i].valleys)==0):
            return line
        # choose the closest valley in right chunk to the start valley
        connected_to = -1
        min_dist = sys.maxsize
        # found valleys's size in this chunk
        valleys_size = len(self.chunks[i].valleys)
        for j in range(valleys_size):
            valley = self.chunks[i].valleys[j]
            # Check if the valley is not connected to any other valley
            if valley.used is True:
                continue

            dist = abs(current_valley.position - valley.position)

            if min_dist > dist and dist <= valleys_min_abs_dist:
                min_dist = dist
                connected_to = j

        # Return line if the current valley is not connected any more to a new valley in the current chunk of index i.
        if connected_to == -1:
            return line

        line.valley_ids.append(self.chunks[i].valleys[connected_to].valley_id)
        v = self.chunks[i].valleys[connected_to]
        v.used = True
        return self.connect_valleys(i - 1, v, line, valleys_min_abs_dist)

    def get_initial_lines(self):
        number_of_heights = 0
        valleys_min_abs_dist = 0

        # Get the histogram of the first CHUNK_TO_BE_PROCESSED and get the overall average line height.
        for i in range(CHUNK_TOBE_PROCESSED):
            self.avg_height = self.chunks[i].find_peaks_valleys(self.map_valley)

            if self.avg_height:
                number_of_heights += 1
            valleys_min_abs_dist += self.avg_height

        valleys_min_abs_dist /= number_of_heights
        print("Estimated avg line height {}".format(valleys_min_abs_dist))
        self.predicted_line_height = valleys_min_abs_dist

        # Start from the CHUNK_TOBE_PROCESSED chunk
        for i in range(CHUNK_TOBE_PROCESSED - 1, 0, -1):
            if len(self.chunks[i].valleys) == 0:
                continue

            # Connect each valley with the nearest ones in the left chunks.
            for valley in self.chunks[i].valleys:
                if valley.used is True:
                    continue

                # Start a new line having the current valley and connect it with others in the left.
                valley.used = True

                new_line = Line(valley.valley_id)
                new_line = self.connect_valleys(i - 1, valley, new_line, valleys_min_abs_dist)
                new_line.generate_initial_points(self.chunk_width, self.img.shape[1], self.map_valley)

                if len(new_line.valley_ids) > 1:
                    self.initial_lines.append(new_line)
        print("There are {}".format(len(self.initial_lines)) + " initial lines")

    def generate_regions(self):
        if len(self.initial_lines) == 0:
            return
        # Sort lines by row position
        self.initial_lines.sort(key=Line().get_min_row_position)
        #reset line regions
        self.lines_region = []
        # Add first region
        r = Region(top=Line(), bottom=self.initial_lines[0])

        # r.update_region(self.thresh, 0)
        r.update_region(self.gray_img, 0)
        self.initial_lines[0].above = r
        self.lines_region.append(r)

        if r.height < self.predicted_line_height * 2.5:
            self.avg_line_height += r.height

        # Add rest of regions.
        for i in range(len(self.initial_lines)):
            top_line = self.initial_lines[i]
            if i + 1 < len(self.initial_lines):
                bottom_line = self.initial_lines[i + 1]
            else:
                bottom_line = Line()
            # Assign lines to region.
            r = Region(top_line, bottom_line)
            # res = r.update_region(self.thresh, i)
            res = r.update_region(self.gray_img, i)
            # Assign regions to lines
            if top_line.initial_valley_id != -1:
                top_line.below = r

            if bottom_line.initial_valley_id != -1:
                bottom_line.above = r
            if res is False:
                self.lines_region.append(r)
                if (r.height < self.predicted_line_height * 2.5):
                    self.avg_line_height += r.height

        if len(self.lines_region) > 0:
            self.avg_line_height /= len(self.lines_region)
            print("Avg line height is {}".format(int(self.avg_line_height)))

    def component_belongs_to_above_region(self, line, contour):
        # Calculate probabilities
        probAbovePrimes = [0] * len(self.primes)
        probBelowPrimes = [0] * len(self.primes)
        n = 0

        tl = (contour[0], contour[1])  # top left

        width, height = contour[2], contour[3]

        for i_contour in range(tl[0], tl[0] + width):
            for j_contour in range(tl[1], tl[1] + height):
                if self.thresh[j_contour][i_contour] == 255:
                    continue

                n += 1

                contour_point = np.zeros([1, 2], dtype=np.uint8)
                contour_point[0][0] = j_contour
                contour_point[0][1] = i_contour
                # print(contour_point)
                if line.above != 0:
                    newProbAbove = int(line.above.bi_variate_gaussian_density(
                        contour_point))
                else:
                    newProbAbove = 0

                if line.below != 0:
                    newProbBelow = int(line.below.bi_variate_gaussian_density(
                        contour_point))
                else:
                    newProbBelow = 0

                self.addPrimesToList(newProbAbove, probAbovePrimes)
                self.addPrimesToList(newProbBelow, probBelowPrimes)
                # print(newProbAbove, newProbBelow)

        prob_above = 0
        prob_below = 0

        for k in range(len(probAbovePrimes)):
            mini = min(probAbovePrimes[k], probBelowPrimes[k])

            probAbovePrimes[k] -= mini
            probBelowPrimes[k] -= mini

            prob_above += probAbovePrimes[k] * self.primes[k]
            prob_below += probBelowPrimes[k] * self.primes[k]


        return prob_above < prob_below, line, contour

    def repair_lines(self):
        """
        repeair all initial lines and generate the final line region
        """

        for line in self.initial_lines:
            column_processed = {}  # int, bool

            for column in range(self.img.shape[1]):#cols
                column_processed[column] = False

            i = 0
            while i < len(line.points):
                point = line.points[i]
                x = int(point[0])
                y = int(point[1])
                # print(y)
                # Check for vertical line intersection
                # In lines, we don't save all the vertical points we save only the start point and the end point.
                # So in line->points all we save is the horizontal points so, we know there exists a vertical line by
                # comparing the point[i].x (row) with point[i-1].x (row)
                if self.thresh[x][y] == 255:
                    if i == 0:
                        i+=1
                        continue
                    black_found = False
                    if line.points[i - 1][0] != line.points[i][0]:
                        # Means the points are in different rows (a vertical line).
                        min_row = int(min(line.points[i - 1][0], line.points[i][0]))
                        max_row = int(max(line.points[i - 1][0], line.points[i][0]))

                        for j in range(int(min_row), int(max_row) + 1):
                            if black_found is True:
                                break
                            if self.thresh[j][line.points[i - 1][1]] == 0:
                                x = j
                                y = line.points[i - 1][1]
                                black_found = True

                    if black_found == False:
                        i+=1
                        continue

                # Ignore it's previously processed

                if column_processed[y] == True:
                    i+=1
                    continue

                # Mark column as processed
                column_processed[y] = True

                self.avg_line_height = int(self.avg_line_height)

                for c in self.contours:
                    # Check line & contour intersection
                    tl = (c[0], c[1])
                    br = (c[0] + c[2], c[1] + c[3])

                    if y >= tl[0] and y <= br[0] and x >= tl[1] and x <= br[1]:
                        # print("br: {}".format(br))
                        # If contour is longer than the average height ignore
                        height = br[1] - tl[1]

                        if height > int(self.avg_line_height * 0.9):
                            continue

                        is_component_above, line, c = self.component_belongs_to_above_region(line, c)

                        # print(is_component_above)
                        new_row = 0
                        if is_component_above == False:
                            new_row = tl[1]
                            line.min_row_pos = min(line.min_row_pos, new_row)
                        else:
                            new_row = br[1]
                            line.max_row_pos = max(new_row, line.max_row_pos)

                        width = c[2]

                        for k in range(tl[0], tl[0] + width):
                            point_new = (new_row, line.points[k][1]) # make this coz tuple does not have value-assignment
                            line.points[k] = point_new

                        i = br[0]  # bottom right

                        # print("I: {}".format(i))
                        break  # Contour found
                i += 1

    def get_regions(self):
        ret = []
        for region in self.lines_region:
            ret.append(region.region.copy())
        return ret

    def save_image_with_lines(self, path):
        img_clone = self.img.copy()

        for line in self.initial_lines:
            last_row = -1

            for point in line.points:
                img_clone[int(point[0])][int(point[1])] = (0, 0, 255)
                # Check and draw vertical lines if found.
                if last_row != -1 and point[0] != last_row:
                    for i in range(min(int(last_row), int(point[0])), max(int(last_row), int(point[0]))):
                        img_clone[int(i)][int(point[1])] = (0, 0, 255)

                last_row = point[0]

        # cv2.imwrite(path, img_clone)

    def save_lines_to_file(self, lines):
        """
        lines: list contains multiple images as numpy arrays
        Return pathes containing various output image pathes
        """
        output_image_path = []
        idx = 0
        if len(self.initial_lines) == 0:
            path = self.output_path + "/Line_" + str(idx) + ".jpg"
            output_image_path.append(path)
            cv2.imwrite(path, self.img)
            return output_image_path

        idx = 0
        output_image_path = []
        for m in lines:
            path = self.output_path + "/Line_" + str(idx) + ".jpg"
            cv2.imwrite(path, m)
            output_image_path.append(path)
            idx += 1

        return output_image_path

    def segment(self):
        """
        return a list of cut images's path
        """
        self.pre_process_img()
        self.find_contours()
        # Divide image into vertical chunks
        self.generate_chunks()
        # Get initial lines
        self.get_initial_lines()

        self.save_image_with_lines(self.output_path + "/Initial_Lines.jpg")

        # Get initial line regions
        self.generate_regions()
        # repeair all initial lines and generate the final line region
        self.repair_lines()
        # Generate the final line regions
        self.generate_regions()

        # save image drawn lines
        self.save_image_with_lines(self.output_path + "/Final_Lines.bmp")

        return self.get_regions()
