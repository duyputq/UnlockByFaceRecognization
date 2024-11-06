#File: classifier.py
#Desciption: Train anh cho model haar cascade roi xuat ra file cassifier.xml
import numpy as np
from PIL import  Image
import os, cv2

# Method to train custom classifier to recognize face
def train_classifer(data_dir):
    """
    Train clf de nhan ra khuon mat cua user
    :param data_dir: thu muc luu
    :return: file xml
    """
    # Doc tat ca anh tu thu muc data
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    # Luu anh duoi dang numpy va id cua user = index cua imageNP
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train va xuat ra file
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

def main():
    train_classifer("data")

if __name__ == "__main__":
    main()