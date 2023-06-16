"""
Represent the separator among line regions.
"""
CHUNK_NUMBER = 20

class Line():
    def __init__(self, initial_valley_id  = -1):
        """
        min_row_pos : the row at which a region starts
        max_row_pos : the row at which a region ends
        
        above: Region above the line
        below: Region below the line

        valley_ids: ids of valleys
        points: points representing the line
        """
        
        self.min_row_pos = 0
        self.max_row_pos = 0
        self.points = [] # (x,y)


        self.above = 0 #Region type
        self.below = 0 #Region type
        self.valley_ids = []
        if initial_valley_id != -1: #means that there is a valley
            self.valley_ids.append(initial_valley_id)
        self.initial_valley_id = initial_valley_id
    
    def generate_initial_points(self, chunk_width, img_width, map_valley={}):
        c = 0
        prev_row = 0
        #sort the valleys according to their chunk number
        self.valley_ids.sort()
        #add line points in the first chunks having no valleys
        if map_valley[self.valley_ids[0]].chunk_index > 0:
            prev_row = map_valley[self.valley_ids[0]].position
            self.max_row_pos = self.min_row_pos = prev_row
            for j in range(map_valley[self.valley_ids[0]].chunk_index * chunk_width):
                if c == j:
                    c += 1
                    self.points.append((prev_row,j))

        # Add line points between the valleys
        for id in self.valley_ids:
            chunk_index = map_valley[id].chunk_index
            chunk_row = map_valley[id].position
            chunk_start_column = chunk_index * chunk_width

            for j in range(chunk_start_column, chunk_start_column + chunk_width):
                self.min_row_pos = min(self.min_row_pos, chunk_row)
                self.max_row_pos = max(self.max_row_pos, chunk_row)
                if c == j:
                    c+=1
                    self.points.append((chunk_row, j))
        
            if prev_row != chunk_row:
                prev_row = chunk_row
                self.min_row_pos = min(self.min_row_pos, chunk_row)
                self.max_row_pos = max(self.max_row_pos, chunk_row)
        # Add line points in the last chunks having no valleys
        if CHUNK_NUMBER - 1 > map_valley[self.valley_ids[-1]].chunk_index:
            chunk_index = map_valley[self.valley_ids[-1]].chunk_index
            chunk_row = map_valley[self.valley_ids[-1]].position

            for j in range(chunk_index * chunk_width + chunk_width,img_width):
                if c == j:
                    c+=1
                    self.points.append((chunk_row, j))

        
    def get_min_row_position(self, line):
        return line.min_row_pos


