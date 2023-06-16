"""
A class representing peaks (local maximum points in a histogram)
"""
class Peak:
    def __init__(self, position=0, value=0):
        """
        position: row position
        value: the number of foreground pixels
        """
        self.position = position
        self.value = value

    def get_value(self, peak):
        """
        used to sort Peak lists based on value
        """
        return peak.value
    def get_row_position(self, peak):
        """
        used to sort Peak lists based on row position
        :param peak:  
        """
        return peak.position
