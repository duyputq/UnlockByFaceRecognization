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