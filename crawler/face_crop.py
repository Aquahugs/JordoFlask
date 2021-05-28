import dlib
import cv2
import argparse 

def crop_face(input, output):
    print('crop', input, output)
    img = cv2.imread(input)
    h, w, _ = img.shape
    
    detector = dlib.get_frontal_face_detector()

    dets = detector(img, 1)

    for i, face in enumerate(dets):
        width = face.right() - face.left()
        height = face.bottom() - face.top()

        left = face.left() - width // 2 
        right = face.right() + width // 2 
        top = face.top() - height // 2
        bottom = face.bottom() + height // 2
        
        if left < 0: left = 0
        if right > w: right = w
        if top < 0: top = 0 
        if bottom > h: bottom = h 

        crop = img[top:bottom, left:right]
        crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output, crop)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image')
    parser.add_argument('output', help='output')
    args = parser.parse_args()

    crop_face(args.input, args.output)