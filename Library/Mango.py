import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import signal
import math
from scipy import signal
from Library import ProLib

class Mango:
    # Hàm khởi tạo - Khởi tạo các biến cần thiết
    # @ configPath: Đường dẫn đến file config
    def __init__(self, configPath):
        # Đọc file config
        f = open(configPath)
        data = json.load(f)

        # Khởi tạo các biến chung
        self.__roi = [] # Mảng chứa khung hình vùng quan tâm.
        self.__rmBG = [] # Mảng chứa khung ảnh xoài đã được xóa nền.
        self.__area = [] # Mảng chứa diện tích bề mặt xoài qua từng khung hình.
        self.__fg = [] # Mảng chứa ảnh nhị phân tiết diện bề mặt xoài qua từng khung hình.
        
        # Khởi tạo các biến đặc biệt
        self.__roiSize = data["ROI Size"] # Kích thước vùng quan tâm.
        self.__trackThresh = data["Tracking Thresh"] # Ngưỡng sử dụng để tìm vùng quan tâm.
        self.__rotated = data["Rotated"] # Biến xác định có cần xoay video 90 độ không
        # Khởi tạo mặt nạ LAB
        self.__lowerLAB = np.array(data["LAB Filter"]["Lower"])
        self.__upperLAB = np.array(data["LAB Filter"]["Upper"])
        self.__minArea = data["Min Area"] # Giới hạn diện tích vết khiếm khuyết
        # Đóng file json
        f.close()
        
    # Kiểm tra xem có cần phải xoay video 90 độ không
    def need2Rotate(self):
        return True if self.__rotated == 1 else False

    # Xác định vị trí xoài
    # @ frame: Khung hình gốc
    # @ output: trackedMango - ROI
    #           inWorkingArea - True nếu xoài trong vùng làm việc và ngược lại
    #           roiCoordinate - Tọa độ ROI
    def trackingMango(self, frame):
        # Khởi tạo
        threshCont = self.__trackThresh
        scaleRate = 0.125

        # Xác định kích thước vùng quan tâm
        h, w = self.__roiSize

        # Xác định kích thước khung hình gốc
        H, W = frame.shape[:2]

        # Trích kênh xanh lá
        grayFrame = frame[:,:,1]

        # Lấy ngưỡng
        _, thresh = cv2.threshold(grayFrame, threshCont, 255, cv2.THRESH_BINARY)

        # Thu nhỏ ảnh ngưỡng
        scaledThresh = ProLib.scale(thresh, scaleRate) # Thu nhỏ 1:8

        # Khử nhiễu
        kernel = np.ones((5,5), np.uint8)
        scaledThresh = cv2.morphologyEx(scaledThresh, cv2.MORPH_OPEN, kernel)
        scaledThresh = cv2.morphologyEx(scaledThresh, cv2.MORPH_CLOSE, kernel)

        # Xác định tâm ROI
        M = cv2.moments(scaledThresh)
        if  M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])*(1/math.sqrt(scaleRate))
            cY = int(M["m01"] / M["m00"])*(1/math.sqrt(scaleRate))
            # Khi thuật toán moments xác định được tâm ROI -> Xoài trong vùng làm việc
            inWorkingArea = True
        
        else:
            cX = W
            cY = H/2

        # Xác định xoài trong vùng làm việc
        # Rìa trái của ROI không bị cắt bởi khung hình
        if cX - w/2 > 0:
            xLeft = int(cX - w/2)
        else:
            inWorkingArea = False
            xLeft = 0
        # Rìa phải của ROI không bị cắt bởi khung hình
        if cX + w/2 < W:
            xRight = int(cX + w/2)
        else:
            inWorkingArea = False
            xRight = W
        
        # Rìa trên của ROI không bị cắt bởi khung hình
        if cY - h/2 > 0:
            yUpper = int(cY - h/2)
        else:
            yUpper = 0

        # Rìa dưới của ROI không bị cắt bởi khung hình
        if cY + h/2 < H:
            yLower = int(cY + h/2)
        else:
            yLower = H
        
        # Tách ROI ra khỏi khung hình
        trackedMango = frame[yUpper : yLower, xLeft : xRight]

        # Tọa độ điểm đầu, cuối của ROI
        roiCordinate = [xLeft, yUpper, xRight, yLower]
        
        # Lưu ROI khi xoài trong vùng làm việc
        if inWorkingArea:
            self.__roi.append(trackedMango)

        return trackedMango, inWorkingArea, roiCordinate

    # Tách xoài ra khỏi nền
    # @ inWorkingArea: Nhận từ method trackingMango. True khi xoài trong vùng làm việc, ngược lại False
    # @ output: rmbgFrame - Khung hình xoài đã được tách ra khỏi nền
    #           cropMask  - Mặt nạ tách xoài, cũng là ảnh nhị phân tiết diện bề mặt xoài
    def rmBackground(self, inWorkingArea):
        # Khởi tạo
        # Chỉ tiến hành xử lý khi xoài trong vùng làm việc
        if inWorkingArea:
            frame = self.__roi[-1]

            # Khử nhiễu
            blurFrame = cv2.blur(frame, (5,5))

            # Chuyển hệ màu của frame từ RGB sang Gray
            frameLAB = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2LAB)

            # Lấy ảnh nhị phân bằng filter LAB
            threshLAB = cv2.inRange(frameLAB, self.__lowerLAB, self.__upperLAB)
            
            # Trích xuất ảnh nhị phân tiết diện bề mặt xoài để tạo mặt nạ crop xoài
            cont, _ = cv2.findContours(threshLAB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cropMask = np.zeros_like(threshLAB)

            for cnt in cont:
                if cv2.contourArea(cnt) > self.__minArea:
                    cv2.drawContours(cropMask, [cnt], 0, 255, -1)

            rmbgFrame = cv2.bitwise_and(frame, frame, mask=cropMask)

            # Tính toán và lưu trữ
            self.__rmBG.append(rmbgFrame)
            self.__area.append(cv2.countNonZero(cropMask))
            self.__fg.append(cropMask)

            return rmbgFrame, cropMask

        else:
            # Trả về ảnh trống khi xoài ngoài vùng làm việc
            blank = np.zeros((self.__roiSize[0], self.__roiSize[1], 3), np.uint8)
            return blank, blank

    # Xác định 4 mặt xoài
    # showFigure: True khi cần show biểu đồ, ngược lại False
    # figurePath: Đường dẫn lưu biểu đồ về diện tích bề mặt xoài
    # qua từng khung hình. Nếu không truyền đường dẫn vào, phương thức này chỉ 
    # xuất biểu đồ ra màn hình mà không lưu lại. Tên biểu đồ cần kêt thúc bằng .png hoặc .jpg
    # output: bigIdx    - Chỉ số mảng của 2 mặt cạnh
    #         smallIdx  - Chỉ số mảng của mặt lưng và bụng
    def detectMainFaces(self, showFigure = False, figurePath=None):
        # Tính giá trị trung bình động của bộ dữ liệu về diện tích bề mặt xoài trong từng khung hình
        movingAvg = []
        window = 7
        for i in range(len(self.__area) - window + 1):
            movingAvg.append(np.mean(self.__area[i:i+window]))

        # Fit dữ liệu trung bình động với dữ liệu gốc
        # Đệm NAN vào những ô không chứ giá trị trong mảng chứa dữ liệu trung bình động
        for i in range(int((window - 1)/2)):
            movingAvg.insert(0, np.nan)
        for i in range(int((window - 1)/2)):
            movingAvg.append(np.nan)

        # Giá trị trung bình của dữ liệu trung bình động
        avg = np.nanmean(movingAvg)

        # Trích xuất chỉ số mảng của 2 mặt cạnh từ các cực trị của dữ liệu trung bình động
        bigIdx, _ = signal.find_peaks(movingAvg, height=avg, distance=20)
        # Lật ngược dữ liệu trung bình động và tìm cực trị để tìm được chỉ số mảng của mặt lưng và bụng
        mavgFrameFlip = [elements * -1 for elements in movingAvg] 
        smallIdx, _ = signal.find_peaks(mavgFrameFlip, height=-avg, distance=20)

        # Xử lý dữ liệu
        # Khi thu được nhiều hơn 2 khung hình mặt cạnh hoặc 2 mặt lưng, bụng
        # Bỏ bớt các khung hình nằm ngoài rìa của mảng dữ liệu
        # Khi không xác định đủ số mặt, thuật toán sẽ in thông báo lên màn monitor
        # đồng thời thêm đệm thêm dữ liệu ảo vào vùng dữ liệu còn thiếu để tránh crash
        isProcessed = False
        while not isProcessed:
            if len(bigIdx) > 2:
                if bigIdx[0] - 0 > (len(self.__area) - 1) - bigIdx[-1]:
                    bigIdx = bigIdx[:2]
                else:
                    bigIdx = bigIdx[1:]
            elif len(bigIdx) < 2:
                print("Cannot determine 4 faces")
                bigIdx.append(bigIdx[0])
            
            if len(smallIdx) > 2:
                if smallIdx[0] - 0 > (len(self.__area) - 1) - smallIdx[-1]:
                    smallIdx = smallIdx[:2]
                else:
                    smallIdx = smallIdx[1:]
            elif len(smallIdx) < 2:
                print("Cannot determine 4 faces")
            
            if (len(bigIdx) == 2) & (len(smallIdx) == 2):
                isProcessed = True

        # Vẽ đồ thị biểu diễn diện tích của bề mặt xoài qua từng khung hình
        x = list(range(0, len(self.__area)))

        plt.figure(figsize=(10,7))
        plt.scatter(x, self.__area, marker='o', color="green") # Biểu đồ diện tích
        plt.plot(x, movingAvg, color='orange') # Biểu đồ trung bình động

        # Các điểm cực trị
        plt.plot(bigIdx[0], movingAvg[bigIdx[0]], marker='o', color="red")
        plt.plot(smallIdx[0], movingAvg[smallIdx[0]], marker='o', color="blue")
        plt.plot(bigIdx[1], movingAvg[bigIdx[1]], marker='o', color="red")
        plt.plot(smallIdx[1], movingAvg[smallIdx[1]], marker='o', color="blue")

        # Cài đặt và chú thích thêm cho biểu đồ
        plt.ylim(min(self.__area)-500, max(self.__area)+2500)
        plt.grid()

        # Đặt tên các trục tọa độ
        plt.xlabel("Khung hình (Frame)", fontsize=12)
        plt.ylabel("Diện tích (Pixel)", fontsize=12)

        # Thêm chú thích
        plt.legend(["Dữ liệu diện tích thô", "Dữ liệu trung bình động (5 khung hình liên tiếp)", "Điểm cực đại", "Điểm cực tiểu"], ncol=2, fontsize=10, loc="upper right")

        # Tiêu đề biểu đồ
        plt.title("Diện tích bề mặt xoài", fontsize=16)

        # Lưu biểu đồ
        if not figurePath == None:
            plt.savefig(figurePath)

        # Xuất biểu đồ ra màn hình
        if showFigure == True:
            plt.show()
        plt.close()

        # Xác định các mặt chính của quả xoài
        bigFace1 = self.__rmBG[bigIdx[0]]
        bigFace2 = self.__rmBG[bigIdx[1]]
        smallFace1 = self.__rmBG[smallIdx[0]]
        smallFace2 = self.__rmBG[smallIdx[1]]

        return (bigFace1, bigFace2), (smallFace1, smallFace2)