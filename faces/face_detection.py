import dlib
import cv2

def get_face_position(img_src):
    img = cv2.imread(img_src)
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

        return True, {
            'top': top,
            'bottom': bottom,
            'right': right,
            'left': left,
        }

    return False, {
        'top': None,
        'bottom': None,
        'right': None,
        'left': None,
    }

def crop_face(img, face_position, width=None, height=None, output_image=None):
    top = face_position['top']
    bottom = face_position['bottom']
    right = face_position['right']
    left = face_position['left']

    crop = img[top:bottom, left:right]
    h, w, _ = img.shape

    if width is None:
        width = w

    if height is None:
        height = h

    crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)

    if output_image is not None:
        cv2.imwrite(output_image, crop)

    return crop