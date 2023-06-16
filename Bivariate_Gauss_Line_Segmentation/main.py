# %%
from Bivariate_LineSegmentation import LineSegmentation
import cv2
def line_segment(img, output_path):
    """
    img: image needed to segment to lines.
    output_path: output folder of segmented images. 
    :return: output image pathes
    """
    # cv2.imshow("Original", img)
    # cv2.waitKey()
    # (1) Segment Lines
    line_segmentation = LineSegmentation(img=img, output_path=output_path)
    lines = line_segmentation.segment()
    # (2) Save lines to file
    output_image_path = line_segmentation.save_lines_to_file(lines)

    return output_image_path
    
# %%
img = cv2.imread("../test/7.jpg")
img.shape

# %%
#do program
output_path = "result"
r =line_segment(img, output_path)
print(r)



# %%
