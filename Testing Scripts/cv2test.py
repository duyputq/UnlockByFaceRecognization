import cv2

image = cv2.imread('C:/Users/Duy/Pictures/tiger.jpg')

if image is None:
    print("Error:.")
else:
    image = cv2.resize(image, (600, 600))
    cv2.imshow('Hinh Anh', image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()