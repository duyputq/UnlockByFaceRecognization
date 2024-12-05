#File: main.py
#Description: file nay nhan dien khuon mat theo du lieu da train tu file classifier.xml
import cv2

def draw_boundary(img, classfier, scaleFactor, minNeighbors, color, text, clf):
    """
    ham set up toa do cho duong bao khuon mat
    :param img: anh ban dau
    :param classfier: doi tuong phan loai
    :param scaleFactor: dieu chinh scale (float)
    :param minNeighbors: so luong lang gieng
    :param color: mau RGB
    :param text: van ban hien thi ten nguoi
    :param clf: bien tu cv2.face.LBPHFaceRecognizer_create()
    :return: coords (toa do khuon mat)
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classfier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        id, confidence =clf.predict(gray_img[y:y+h, x:x+w])
        print("Id khuon mat: ",id)
        if confidence > 50:
            cv2.putText(img, "Stranger", (x,y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA )
        elif id == 2:
            cv2.putText(img, "Dinh Viet Hieu", (x,y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA )
        coords = [x,y,w,h]
    return  coords

def recognize(img, clf, faceCascade):
    """
    ham render ra o vuong khuon mat
    :param img: anh khuon mat tu data
    :param clf: cv2.face_LBPHFaceRecognizer - phan loai cua openCV
    :param faceCascade:  cv2.CascadeClassifier - phan loai cua haar cascade
    :return:
    """
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img

def main():
    """
    ham main
    :return:
    """
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


