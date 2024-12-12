import cv2
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
        mask[y:y+h, x:x+w] = 255

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
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
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

def keep_red_on_face(image, face_regions, red_threshold=150, color_diff=50):
    """
    Retain red points on the face and darken other areas.
    """
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]

    # Create a strict red mask
    red_mask = (
        (red_channel > red_threshold) &
        (red_channel > green_channel + color_diff) &
        (red_channel > blue_channel + color_diff)
    ).astype(np.uint8)

    # Create a face mask
    face_mask = np.zeros_like(red_mask)
    for (x, y, w, h) in face_regions:
        face_mask[y:y+h, x:x+w] = 1  # Mark the face region

    # Combine masks to retain red points only within the face region
    combined_mask = cv2.bitwise_and(red_mask, face_mask)

    # Lọc các vùng lớn hoặc quá lệch
    refined_mask = filter_regions_by_shape(combined_mask, cv2.findNonZero(combined_mask), max_deviation=5)

    # Lấy tọa độ các điểm đỏ
    points = cv2.findNonZero(refined_mask)
    return refined_mask, points

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
        distance = abs(m * x - y + b) / np.sqrt(m**2 + 1)
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

    # Display results
    result_image = cv2.bitwise_and(image, image, mask=red_mask)
    cv2.imshow("Original Image", image)
    cv2.imshow("Red Points on Face", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return avg_deviation, variance

def main():
    # Load input image
    image = cv2.imread("test2.png")
    if image is None:
        print("Failed to load image.")
        return

    # Process laser image
    avg_deviation, variance = process_laser_image(image)
    if avg_deviation is None:
        print("Laser line not detected!")
    else:
        print(f"Average Deviation: {avg_deviation}, Variance: {variance}")
        if avg_deviation >= 1.5 and variance >= 1:
            print("3D face detected (Real Person)")
        else:
            print("2D face detected (Photo)")

if __name__ == "__main__":
    main()
