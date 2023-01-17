import cv2
import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.getcwd())

from Library import ProLib

'''Chương trình được xây dựng nhằm tính diện tích trung bình của mỗi điểm ảnh
trong các ô cờ của ảnh bàn cờ đầu vào.'''


# Khởi tạo biến và đường dẫn
# Truyền vào đường dẫn ảnh bàn cờ đã được xử lý bóp méo, biến dạng
imgPath = "Pixel Area/Checkerboard Nov_22/Undistort Checkerboard"
imgName = "Checkerboard_80.png"

# Đọc ảnh đầu vào
rawCB = cv2.imread(f"{imgPath}/{imgName}")

# Xử lý ảnh bàn cờ
rawCB = cv2.GaussianBlur(rawCB, (5,5), 0)
grayCB = cv2.cvtColor(rawCB, cv2.COLOR_BGR2GRAY)
_, threshCB = cv2.threshold(grayCB, 100, 255, cv2.THRESH_BINARY)

# Làm mượt cạnh bàn cờ
kernel = np.ones((3,3), np.uint8)
prCB = cv2.morphologyEx(threshCB, cv2.MORPH_CLOSE, kernel, iterations=1)

# Xác định và vẽ contours
cont, hier = cv2.findContours(prCB, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
totalArea = 0
noSquare = 0
for cnt in cont:
    area = cv2.contourArea(cnt)
    if area > 1000 and area < 8000:
        noSquare += 1
        cv2.drawContours(rawCB, [cnt], 0, (0, 0, 255), 1)
        print(f"Square {noSquare}: ", area)
        totalArea += area
        cv2.imshow("Contours", rawCB)

# Tính diện tích trung bình của mỗi điểm ảnh
noPixel = round(totalArea/noSquare, 2)
pixelArea = round((400/noPixel), 4)

# Xuất kết quả ra màn hình
print("Number of square: ", noSquare)
print("NOP/Square: ", noPixel)
print("Pixel Area: ", pixelArea)

cv2.waitKey()
cv2.destroyAllWindows()