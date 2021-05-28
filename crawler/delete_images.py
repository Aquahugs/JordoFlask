import os

edge_dir = 'images/edges/train'
face_dir = 'images/faces/train'

edges = os.listdir(edge_dir)
faces = os.listdir(face_dir)

for file in faces:
    print(file)
    if file not in edges:
        try:
            os.remove(face_dir + '/' + file)
        except error:
            print(err)