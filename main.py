#File: main.py
#Description: file nay nhan dien khuon mat theo du lieu da train tu file classifier.xml
import cv2
import time
import numpy as np

def detect_face_region(image):
    """
    Detect faces in the image and return their coordinates.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def remove_background(image, face_regions):
    """
    Remove the background outside the detected face regions.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in face_regions:
        mask[y:y + h, x:x + w] = 255

    # Refine the mask to ensure only the face region remains
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)

    # Apply Gaussian blur to smooth transitions
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def filter_regions_by_shape(mask, points, max_deviation=5, max_size=5000, min_size=10):
    """
    Lọc các mảng lớn hoặc quá lệch so với đường xu hướng.
    """
    # Phân tích các thành phần kết nối (connected components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Tạo mặt nạ mới để giữ lại các vùng phù hợp
    refined_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # Bỏ qua background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
            i, cv2.CC_STAT_HEIGHT]

        # Loại bỏ các mảng quá lớn hoặc quá nhỏ
        if area < min_size or area > max_size:
            continue

        # Lấy tất cả các điểm trong vùng này
        component_mask = (labels == i).astype(np.uint8)
        component_points = cv2.findNonZero(component_mask)

        # Tính đường xu hướng (linear regression)
        m, b = fit_trend_line(component_points)

        # Kiểm tra độ lệch của từng điểm trong vùng
        avg_deviation, _ = calculate_average_deviation(component_points, m, b)
        if avg_deviation > max_deviation:
            continue  # Bỏ qua nếu độ lệch quá lớn

        # Giữ lại vùng này trong mặt nạ mới
        refined_mask = cv2.bitwise_or(refined_mask, component_mask)

    return refined_mask

#
# def keep_red_on_face(image, face_regions, red_threshold=200, color_diff=50):
#     """
#     Retain red points on the face and darken other areas.
#     """
#     red_channel = image[:, :, 2]
#     green_channel = image[:, :, 1]
#     blue_channel = image[:, :, 0]
#
#     # Create a strict red mask
#     red_mask = (
#             (red_channel > red_threshold) &
#             (red_channel > green_channel + color_diff) &
#             (red_channel > blue_channel + color_diff)
#     ).astype(np.uint8)
#
#     # Create a face mask
#     face_mask = np.zeros_like(red_mask)
#     for (x, y, w, h) in face_regions:
#         face_mask[y:y + h, x:x + w] = 1  # Mark the face region
#
#     # Combine masks to retain red points only within the face region
#     combined_mask = cv2.bitwise_and(red_mask, face_mask)
#
#     # Lọc các vùng lớn hoặc quá lệch
#     refined_mask = filter_regions_by_shape(combined_mask, cv2.findNonZero(combined_mask), max_deviation=5)
#
#     # Lấy tọa độ các điểm đỏ
#     points = cv2.findNonZero(refined_mask)
#     return refined_mask, points

def keep_red_on_face(image, face_regions, red_threshold=240, color_diff=40, brightness_threshold=200):
    """
    Giữ lại các điểm đỏ trên khuôn mặt và làm đen các vùng khác.
    """
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]

    # Tạo mặt nạ điểm đỏ nghiêm ngặt
    red_mask = (
        (red_channel > red_threshold) &
        (red_channel > green_channel + color_diff) &
        (red_channel > blue_channel + color_diff)
    ).astype(np.uint8)

    # Loại bỏ các điểm có độ sáng quá cao
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_mask = (brightness < brightness_threshold).astype(np.uint8)

    # Kết hợp mặt nạ đỏ với mặt nạ độ sáng
    red_mask = cv2.bitwise_and(red_mask, brightness_mask)

    # Tạo mặt nạ khuôn mặt
    face_mask = np.zeros_like(red_mask)
    for (x, y, w, h) in face_regions:
        face_mask[y:y+h, x:x+w] = 1  # Đánh dấu vùng khuôn mặt

    # Kết hợp để giữ điểm đỏ trong vùng khuôn mặt
    combined_mask = cv2.bitwise_and(red_mask, face_mask)

    # Lấy tọa độ các điểm đỏ
    points = cv2.findNonZero(combined_mask)
    return combined_mask, points


def fit_trend_line(points):
    """
    Calculate the trend line from points (y = mx + b).
    """
    points = points.squeeze()
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    return m, b


def calculate_average_deviation(points, m, b):
    """
    Calculate average deviation and variance of points from the trend line.
    """
    points = points.squeeze()
    distances = []
    for x, y in points:
        distance = abs(m * x - y + b) / np.sqrt(m ** 2 + 1)
        distances.append(distance)
    return np.mean(distances), np.var(distances)


def process_laser_image(image):
    """
    Process the laser image to identify 3D or 2D surfaces.
    """
    # Detect face
    face_regions = detect_face_region(image)
    if len(face_regions) == 0:
        print("No face detected!")
        return None, None

    # Remove background outside face regions
    image_no_bg = remove_background(image, face_regions)

    # Filter red points in the face region
    red_mask, points = keep_red_on_face(image_no_bg, face_regions)
    if points is None:
        print("No red points detected on the face!")
        return None, None

    # Calculate trend line and deviation
    m, b = fit_trend_line(points)
    avg_deviation, variance = calculate_average_deviation(points, m, b)

    # # Display results
    result_image = cv2.bitwise_and(image, image, mask=red_mask)
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Red Points on Face", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return avg_deviation, variance

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
            cv2.putText(img, "Duy Nguyen", (x,y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA )
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
            avg_deviation, variance = process_laser_image(img)
            print(avg_deviation)
            if avg_deviation is None:
                print("Laser line not detected!")
            else:
                print(f"Average Deviation: {avg_deviation}, Variance: {variance}")
                if avg_deviation >= 2.0 and variance >= 1:
                    print("3D face detected (Real Person)")
                else:
                    print("2D face detected (Photo)")
            time.sleep(0.05)
            cv2.imshow("face detection", img)
            img_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


