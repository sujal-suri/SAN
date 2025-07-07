from PIL import Image, ImageOps
import os
import sys


def do_preprocess(in_path, out_path):
    bmp_path = in_path
    img = Image.open(bmp_path).convert("L")  
    img = ImageOps.invert(img)
    threshold = 128
    bw_img = img.point(lambda x: 255 if x > threshold else 0, '1')
    #target_size = (71, 117) there's no need to resize the image only makes the work worse
    resized_img = bw_img
    resized_img.save(out_path)



if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    do_preprocess(in_path, out_path)

