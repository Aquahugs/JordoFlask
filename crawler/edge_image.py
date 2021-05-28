import cv2 
import argparse
import numpy as np

def process_edge_image(input, output):
    print('edge', input, output)
    img = cv2.imread(input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    edges = cv2.Canny(img, ret * 0.5, ret)

    cv2.imwrite(output, 255 - edges)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image')
    parser.add_argument('output', help='output image')
    args = parser.parse_args()

    process_edge_image(args.input, args.output)