# Mô tả
Nhận diện khuôn mặt sử dụng Haar Cascade

# Thư viện yêu cầu
pip install opencv-python  
pip install numpy
pip install pillow

# Thứ tự chạy file 
1. genFacialData.py : để tạo ra dữ liệu khuôn mặt để lưu trong folder data
2. classifier.py : train data cho model
3. main.py: file nhận diện khuôn mặt

# Mô tả file
1. genFacialData
   - capture liên tục ảnh từ camera 
   - 3 hàm con
     - generate_dataset để lưu ảnh capture jpg
     - draw_boundary mảng lưu tọa độ khuôn mặt
     - detect: dùng tọa độ từ draw_boundary để render ra ô vuông bao quanh mặt sau đó dùng generate_dataset để tạo dataset. Mỗi lần chạy file nhớ đổi user_id để tạo dataset cho người riêng biệt 
   - hàm main
     - mở cam 
     - sau đố chạy hàm detect 
     - có thể thêm hàm time.delay(2) trong while True để làm chậm tốc độ cap ảnh (cần import time trc)

2. classifier.py
   - đọc ảnh cap đc từ data rồi write vào file classifier.xml

3. main.py
   - 2 hàm con:
     - draw_boundary giống trong genFacialData. đặt tên người trong cv2.putText mapping theo id đã được train ở hàm genFacial data.
     - recognize: nhận diện vào ghi tên người (dựa trên data dc train)
   - hàm main: 
     - bộ nhận diện faceCascade cv2 đọc từ file haarcascade_frontalface_default.xml
     - mở và chạy hàm recognize để nhận diện

Trong code chưa có nhận diện laser.

Link kết quả: https://www.canva.com/design/DAGVNFJWd7w/16DujYR6xg8Rsc14FaZ8Tg/edit?ui=eyJIIjp7IkEiOnRydWV9fQ

