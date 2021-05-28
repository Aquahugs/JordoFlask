import os

from face_crop import crop_face
from edge_image import process_edge_image

raw_dir = 'images/raw'
images = os.listdir(raw_dir)
images.sort()

for i, image in enumerate(images):
  print(i, image)
  face_dir = 'images/faces'
  edge_dir = 'images/edges'

  raw_image = raw_dir + '/' + image 
  face_image = face_dir + '/' + image
  edge_image = edge_dir + '/' + image

  if os.path.isfile(face_image) and os.path.isfile(edge_image):
    continue

  crop_face(raw_image, face_image)
  if os.path.isfile(face_image):
    process_edge_image(face_image, edge_image)



