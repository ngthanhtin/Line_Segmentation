from Line import Line
class Valley():
    # next available ID
    ID = 0 # static variable
    def __init__(self, chunk_index=0, position=0):
        """
        chunk_index: The index of the chunk in the chunks vector
        #position: The row position
        """

        # The index of the chunk in the chunks vector
        self.chunk_index = chunk_index
        # The row position
        self.position = position

        # The valley id
        self.valley_id = Valley.ID
        # Whether it's used by a line or not
        self.used = False
        # The line to which this valley is connected
        self.line = Line()

        Valley.ID += 1
    
    def compare_2_valley(self, v1, v2):
        #used to sort valley lists based on position
        return v1.position < v2.position


        
        

