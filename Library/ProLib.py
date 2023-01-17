import cv2
import numpy as np
import pandas as pd
import os, sys
from scipy import signal
import math

# Thay đổi tỉ lệ của ảnh gốc.
# Tỉ lệ được tính theo diện tích khung hình (Không phải theo kích thước width, height)
# @ img: Ảnh gốc.
# @ cnt: Hệ số scale.
# @ output: Ảnh sau khi scale.
def scale(img, cnt):
    # Xác định kích thước ảnh gốc (cao, rộng).
    h, w = img.shape[:2]

    # Thay đổi tỉ lệ chiều cao, chiều rộng.
    scaledH = int(h*math.sqrt(cnt))
    scaledW = int(w*math.sqrt(cnt))

    dim = (scaledW, scaledH)
    res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return res

# Chức năng: Xác định tọa độ của các điểm ảnh khác 0.
# @ img: Ảnh gốc (Ảnh xám hoặc ảnh nhị phân).
# @ output: Mảng chứa tọa độ các điểm ảnh khác 0.
def findNonZero(img):
    find_result = cv2.findNonZero(img)
    if find_result is None:
        return []
    return [x[0] for x in find_result]

# Chức năng: Xác định kích thước cao, rộng của bề mặt xoài.
# @ img: Ảnh xoài đã được tách ra khỏi nền (RGB) hoặc ảnh nhị phân của xoài.
# @ output: mangoSize - Kích thước bề mặt xoài (rộng, cao).
#           boundRect - Ảnh gốc với hình chữ nhật bao quanh xoài.
def getMangoSize(img):
        # Kiểm tra loại ảnh.
        if len(img.shape) > 2:
            # Thu ảnh nhị phân tiết diện bề mặt xoài.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 10, 255, 0)
        else:
            pass

        # Xác định kích thước, vị trí hình chữ nhật bao sát quả xoài.
        x, y, w, h = cv2.boundingRect(img)

        # Tạo ảnh để vẽ chữ nhật bao quanh quả xoài.
        boundRect = img.copy()
        boundRect = cv2.cvtColor(boundRect, cv2.COLOR_GRAY2BGR)

        # Vẽ hình chữ nhật bao quanh xoài.
        cv2.rectangle(boundRect, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
        return (w, h), boundRect

# Vẽ vùng quan tâm dựa vào kích thước cho trước
# @ roi: Tọa độ điểm đầu và điểm cuối của vùng quan tâm ((x1, y1), (y1, y2)).
# @ frame: Khung hình cần vẽ vùng quan tâm.
# @ inWorkingArea: Bằng 1 khi ROI nằm trong vùng làm việc, ngược lại bằng 0.
# @ thickness: Độ dày ROI, mặc định bằng 1.
# @ output: Khùng hình gốc với vùng quan tâm đã được vẽ.
# Vùng quan tâm sẽ được vẽ màu xanh khi entire = 1.
# Nếu entire = 0, vùng quan tâm sẽ được vẽ màu đỏ.
def drawROI(frame, roi, inWorkingArea, thickness=1):
    res = frame.copy()
    
    if inWorkingArea:
        roiColor = (0,255,0) # Green
    else:
        roiColor = (0,0,255) # Red

    cv2.rectangle(res, (roi[0], roi[1]), (roi[2], roi[3]), roiColor, thickness)

    return res

# Ghép nhiều ảnh thành 1 ảnh
# @ imgList: Mảng chứa các bức ảnh cần được ghép.
# @ mode: mode = 0 --> Ghép các ảnh thành hàng ngang.
#         mode - 1 --> Ghép các ảnh thành cột dọc.
def stackImages(imgList, mode):
    # Khởi tạo các biến
    width = []
    height = []
    image = []

    # Lưu kích thước của từng ảnh trong mảng.
    for img in imgList:
        width.append(img.shape[1])
        height.append(img.shape[0])
    
    # Lấy các kích thước lớn nhất.
    maxW = max(width)
    maxH = max(height)

    # Kiểm tra chế độ hoạt động.
    for element in imgList:
        if mode == 0: # Ảnh được ghép thành hàng.
            expandingH = maxH-element.shape[0]
            borderImg = cv2.copyMakeBorder(element, expandingH, 0, 0, 0, cv2.BORDER_CONSTANT)
            image.append(borderImg)
        if mode == 1: # Ảnh được ghép thành cột.
            expandingW = maxW-element.shape[1]
            borderImg = cv2.copyMakeBorder(element, 0, 0, expandingW, 0, cv2.BORDER_CONSTANT)
            image.append(borderImg)
    
    # Tiến hành ghép ảnh.
    if mode == 0:
        res = np.hstack(image)
    if mode == 1:
        res = np.vstack(image)

    return res