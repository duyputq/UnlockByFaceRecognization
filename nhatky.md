# coding convention
```python
import os
import sys

def calculate_area(radius):
    """
    Tính diện tích hình tròn từ bán kính.

    Parameters:
    radius (float): Bán kính của hình tròn.

    Returns:
    float: Diện tích hình tròn.
    """
    if radius < 0:
        raise ValueError("Bán kính phải lớn hơn hoặc bằng 0.")
    return 3.14159 * radius * radius
```

# Code mo camera

```python
import cv2

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Khong the mo camera")
else:
    print("Camera da duoc mo thanh cong")
    while True:
        ret, frame = video_capture.read()

        if ret:
            cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
```

# (5/11) 
-cach dung code

file classifier dua du lieu tu data vao file cassifier.xml
5/11: can tach 2 file 

dau "_" trong biến la biến ignore

# 3 file code:

file genFacialData.py
```python
import cv2

def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

def draw_boundary(img, classfier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classfier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text , (x,y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA )
        coords = [x,y,w,h]
    return  coords

def detect(img, faceCascade, img_id):
    color= {"blue": (255,0,0), "red":(0,0,255), "green":(0,255,0),"white":(255,255,255)}
    coords= draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        user_id = 2
        generate_dataset(roi_img, user_id, img_id)
        # coords = draw_boundary(img, eyeCascade, 1.1, 14, color['red'], "Eyes")
        # coords = draw_boundary(img, noseCascade, 1.1, 5, color['green'], "Nose")
        # coords = draw_boundary(img, mouthCascade, 1.1, 20, color['red'], "Mouth")
    return img

def main():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)

    img_id = 0
    if not video_capture.isOpened():
        print("Khong the mo camera")
    else:
        print("Camera da duoc mo thanh cong")
        while True:
            if img_id % 50 == 0:
                print("collected", img_id, "images")
            _, img = video_capture.read()
            img = detect(img, faceCascade, img_id)
            cv2.imshow("face detectio", img)
            img_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

```
file classifier.py
```python
import numpy as np
from PIL import  Image
import os, cv2

# Method to train custom classifier to recognize face
def train_classifer(data_dir):
    # Read all the images in custom data-set
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")


train_classifer("data")
```

file main.py
```python
import cv2

#ham tra ve toa do cua khuon mat
def draw_boundary(img, classfier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classfier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        id, _ =clf.predict(gray_img[y:y+h, x:x+w])
        print("Id khuong mat: ",id)
        if id == 1:
            cv2.putText(img, "Duy", (x,y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA )
        elif id == 2:
            cv2.putText(img, "Suytdeptrai", (x,y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA )
        else:
            cv2.putText(img, "Stranger", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    return  coords

#ham render ra cai o vuong khuon mat
def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img

def main():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    noseCascade = cv2.CascadeClassifier("Nariz.xml")
    mouthCascade = cv2.CascadeClassifier("Mouth.xml")

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    img_id = 0

    if not video_capture.isOpened():
        print("Không thể mở camera")
    else:
        print("Camera đã được mở thành công")
        while True:
            _, img = video_capture.read()
            img = recognize(img, clf, faceCascade)

            cv2.imshow("face detection", img)
            img_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


```

# (6/11)
Code ra 