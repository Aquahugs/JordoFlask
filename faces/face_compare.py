import numpy as np 
import pickle
import os
import re

def load_faces():
    files = os.listdir('face_vectors')
    sorted(files)

    names = []
    vectors = []
    for file in files:
        pickle_name = os.path.basename(file)
        image_name = pickle_name.split('.')[0]
        name = re.split('\d', image_name)[0]
        names.append(name)

        with open('face_vectors/' + file, 'rb') as f:
            vector = pickle.load(f)
        vectors.append(vector)

    return names, np.array(vectors)

def compare_faces(face_vector, vectors, names):
    result = np.linalg.norm(vectors - face_vector)

