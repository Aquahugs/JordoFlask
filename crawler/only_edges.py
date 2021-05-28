import os

from face_crop import crop_face
from edge_image import process_edge_image

face_dir = 'images/faces/train'
images = os.listdir(face_dir)
images.sort()

for i, image in enumerate(images):
  print(i, image)
  edge_dir = 'images/edges/train'

  face_image = face_dir + '/' + image
  edge_image = edge_dir + '/' + image

  if os.path.isfile(edge_image):
    continue

  process_edge_image(face_image, edge_image)



