#File: genFacialData
#Description: tao ra du lieu khuon mat de train
import cv2

def generate_dataset(img, id, img_id):
    """
    tao dataset
    :param img: anh
    :param id: id user
    :param img_id: id anh
    :return: file anh mat cua user
    """
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

def draw_boundary(img, classfier, scaleFactor, minNeighbors, color, text):
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
        cv2.putText(img, text , (x,y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA )
        coords = [x,y,w,h]
    return  coords

def detect(img, faceCascade, img_id):
    """
    ham phat hien khuon mat dua tren haar cascade
    :param img: input anh
    :param faceCascade:  cv2.CascadeClassifier - phan loai cua haar cascade
    :param img_id: id cua anh
    :return: generate anh roi tra ve img
    """
    color= {"blue": (255,0,0), "red":(0,0,255), "green":(0,255,0),"white":(255,255,255)}
    coords= draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        user_id = 1
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
            cv2.imshow("face detection", img)
            img_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()