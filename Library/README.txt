----------------------------------------------- THƯ VIỆN (LIBRARY) -------------------------------------------------------

NOTE:
- Dưới đây là một số ghi chú về chức năng cũng như cách sử dụng của các module trong thư mục này.
- Các module vẫn cần được tối ưu để đạt hiệu năng tốt nhất.

1/ Basler.py
Đây là module dành riêng cho các tác vụ liên quan đến camera Basler. Bao gồm các hàm như sau:
- getCamInfo: Hiển thị thông tin các camera đang kết nối với thiết bị (laptop, máy tính) của người dùng.
- calibCam: Xuất ra các file config nhằm hiệu chỉnh chống bóp méo hình ảnh.
- undistortImg: Sử dụng các file config thu được từ hàm <calibCam()> để chống bóp méo ảnh đầu vào.
- acquireContinuous: Truy cập camera liên tục - Ghi hình.
- captureImage: Chụp ảnh.

2/ ProLib.py
Đây là module chứa các hàm xử lý ảnh cần thiết.
- scale: Thu nhỏ hoặc phóng to ảnh gốc (theo tỉ lệ diện tích).
- findNonZero: Xác dịnh tọa độ của các điểm ảnh khác 0 của ảnh gray hoặc ảnh nhị phân.
- getMangSize: Xác định kích thước (rộng, dài) của xoài trong khung hình đầu vào (Không thể áp dụng hàm với các mẫu xoài
nằm lệch so với trục thẳng đứng).
- drawROI: Vẽ vùng quan tâm (Được sử dụng để highlight vùng quan tâm - vùng chứa xoài).
- stackImages: Ghép các ảnh có kích thước khác nhau thành hàng ngang hoặc dọc.

3/ Mango.py
Đây là module khai báo class Mango. Bao gồm các thuộc tính và phương thức nhằm thu thập khung hình xoài, xử lý tách xoài ra khỏi nền,
xác định 4 mặt xoài.

4/ MangoFace.py
Đây là module khai báo class MangoFace. Bao gồm các thuộc tính và phương thức xử lý hình ảnh trên các mặt chính của quả xoài. Từ đó
xác định và tính được diện tích thực của các vết khiếm khuyết trên bề mặt xoài.

5/ Calib cam
Chứa các file config nhằm hiệu chỉnh camera đối với thệ thống cũ và hệ thống mới.

6/ Config
Chứa các file khởi tạo .json. Các file này khai báo các thông số cần thiết cho quá trình xử lý hình ảnh quả xoài.