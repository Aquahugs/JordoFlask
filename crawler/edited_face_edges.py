import os

from face_crop import crop_face
from edge_image import process_edge_image

raw_dir = 'images/raw'
images = os.listdir(raw_dir)
images.sort()

train_dir = os.listdir('images/faces/train')
test_dir = os.listdir('images/faces/test')

for i, image in enumerate(images):
  print(i, image)
  face_dir = 'images/faces'

  if image in train_dir or image in test_dir:
    continue 

  raw_image = raw_dir + '/' + image 
  face_image = face_dir + '/' + image

  if os.path.isfile(face_image):
    continue

  crop_face(raw_image, face_image)



