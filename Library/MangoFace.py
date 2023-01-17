import cv2
import numpy as np
import pandas as pd
import json

class MangoFace:
    # Khởi tạo
    def __init__(self, face, faceType, configPath):
        # Đọc file config
        f = open(configPath)
        data = json.load(f)
        
        self.__minDef = data["Min Area of Defect"] # Diện tích nhỏ nhất của các vết khiếm khuyết mà thuật toán phát hiện
        self.__roiKernel = np.ones(data["ROI Kernel Size"], np.uint8) # Kernel bào mòn tạo vùng quan tâm xác định khiếm khuyết
        self.__clipLimit = data["ClipLimit Clahe"] # Ngưỡng giới hạn của thuật toán CLAHE
        self.__gridSize = data["GridSize Clahe"] # Kích thước chia nhỏ khung hình của thuật toán CLAHE
        self.__bigFaceBound = data["Big Face Const"] # Ngưỡng trên và dưới của hệ số diện tích đối với mặt cạnh
        self.__smallFaceBound = data["Small Face Const"] # Ngưỡng trên và dưới của hệ số diện tích đối với mặt lưng và bụng
        self.__dispType = data["Display Type"] # Bằng 0 - Show tổng diện tích của các vết khiếm khuyết, bằng 1 - Show diện tích của từng vết
        self.__fontSize1 = data["Font Size 1"] # Font size 1 (Hiển thị diện tích)
        self.__fontSize2 = data["Font Size 2"] # Font size 2 (Hiển thị diện tích)

        self.__face = face # Mặt xoài gốc
        self.__faceType = faceType # Loại mặt xoài
        self.__slices = [] # Mảng lưu các lát cắt bề mặt
        self.__gray = self.__face[:,:,1] # Ảnh gray mặt xoài
        _, self.__fg = cv2.threshold(self.__gray, 5, 255, cv2.THRESH_BINARY) # Ảnh nhị phân mặt xoài
        self.__area = cv2.countNonZero(self.__fg) # Diện tích bề mặt
        self.__slicedFace = np.zeros_like(self.__fg) # Lưu mặt xoài sau khi cắt lát
        self.__intensity = [] # Mảng chứa các hệ số điểm ảnh được gán lên mỗi lát cắt
        self.__defectCont = self.__face.copy() # Ảnh mặt xoài được highlight các vết khiếm khuyết
        self.__defects = [] # Mảng lưu trữ các vết khiếm khuyết đã được cắt lát
        self.__allDefects = np.zeros_like(self.__fg) # Ảnh chứa tất cả các vết khiếm khuyết đã được cắt lát
        self.__smallFaceConst = [] # Mảng chứa các hệ số diện tích đối với mặt lưng và bụng
        self.__bigFaceConst = [] # Mảng chứa các hệ số diện tích đôi với mặt cạnh
        self.__defArea = 0 # Diện tích khiếm khuyết
    
    
    # Cắt lát bề mặt xoài
    # output: slicedFace - Mặt xoài đã được cắt lát
    #         slices     - Mảng chứa các lát cắt
    def slicing(self):
        # Loại bỏ cuống xoài
        rmStemKernel = np.ones((21,21), np.uint8)
        foreground = cv2.morphologyEx(self.__fg, cv2.MORPH_OPEN, rmStemKernel)

        # Lưu lát cắt đầu tiên
        self.__slices.append(foreground)

        # Tiến hành cắt lát
        currentIter = 0 # Số lần bào mòn hiện tại
        desireIter = 2 # Số lần bào mòn mong muốn
        kernel = np.ones((3,3), np.uint8) # Kernel bào mòn
        while cv2.countNonZero(foreground) > 0:
            foreground = cv2.erode(foreground, kernel)
            currentIter += 1
            if currentIter == desireIter:
                currentIter = 0
                desireIter += 2
                self.__slices.append(foreground)

        # Tiến hành gép các lát cắt
        pxInt = 0 # Giá trị điểm ảnh (Gán lên mỗi lát cắt)
        noSlices = len(self.__slices) # Số lát cắt
        for i in range (noSlices):

            # Lát cắt hiện tại
            currentSlice = self.__slices[i]

            # Tính toán hệ số điểm ảnh tương ứng với từng lát cắt
            # Giá trị điểm ảnh được lấy từ 0 đến 250 để tiện chia cho số lát cắt
            pxInt += int(250/noSlices)
            filter = np.ones(foreground.shape, np.uint8)*pxInt

            # Lưu lại các hệ số điểm ảnh được gán cho các lát cắt
            self.__intensity.append(pxInt)

            # Ghép các lát cắt
            if currentSlice is not self.__slices[-1]:

                # Xác định lát cắt tiếp theo
                nextSlice =  self.__slices[i+1]
                
                # Xứ lý các lát cắt
                cutSlice = cv2.bitwise_xor(currentSlice, nextSlice)
                cutSlice = cv2.bitwise_and(cutSlice, filter)
            
            else:
                cutSlice = cv2.bitwise_and(self.__slices[i], filter)
                
            # Lưu lần lượt các lát cắt sau khi được xử lý vào mảng
            self.__slicedFace = cv2.bitwise_or(self.__slicedFace, cutSlice)
            i += 1

        return self.__slicedFace, self.__slices

    # Xác định các vết khiếm khuyết
    # output: defects - Mảng chứa các vết khiếm khuyết đã được cắt lát
    #         allDefects - Ảnh chứa tất cả các vết khiếm khuyết đã được cắt lát
    #         defectCont - Ảnh mặt xoài với các vết khiếm khuyết đã được highlight
    def findDefect(self):
        # Cân bằng ánh sáng - Tăng độ tương phản
        clahe = cv2.createCLAHE(self.__clipLimit, self.__gridSize)
        grayClahe = clahe.apply(self.__gray)

        # Tạo mặt nạ ROI
        # Khiếm khuyết nằm ngoài ROI sẽ bị bỏ qua bởi thuật toán xác định khiếm khuết
        rmStemKernel = np.ones((25,25), np.uint8)
        maskROI = cv2.morphologyEx(self.__fg, cv2.MORPH_OPEN, rmStemKernel)
        maskROI = cv2.erode(maskROI, self.__roiKernel)
        grayROI = cv2.bitwise_and(grayClahe, grayClahe, mask=maskROI)

        # Lấy ngưỡng tự động OTSU
        _, thresh = cv2.threshold(grayROI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        # Highlight các vết khiếm khuyết
        defectCont, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in defectCont:
            cntArea = cv2.contourArea(cnt)
            if cntArea < 10000 and cntArea > self.__minDef:
                # Tạo blank để trích và lứu các vết khiếm khuyết
                defectFG = np.zeros(self.__face.shape[:2], np.uint8)
                
                # Vẽ viền bao các vết khiếm khuyết
                cv2.drawContours(defectFG, [cnt], 0, 255, -1)
                cv2.drawContours(self.__defectCont, [cnt], 0, (0, 0, 255), 1)

                # Cắt lát các vết khiếm khuyết
                slDefect = cv2.bitwise_and(defectFG, self.__slicedFace)

                if cv2.countNonZero(slDefect) > 0:
                    # Save defect into list
                    self.__defects.append(slDefect)

                    # Save all defect in an image
                    self.__allDefects = cv2.bitwise_or(self.__allDefects, slDefect)

        return self.__defects, self.__allDefects, self.__defectCont

    # Tính toán hệ số diện tích
    def constK(self):
        # Số lát cắt bề mặt xoài
        noSlices = len(self.__slices)
        # Kiểm tra phân loại mặt xoài
        if self.__faceType == 0: # 2 Mặt cạnh
            self.__bigFaceConst.append(self.__bigFaceBound[1])
            const = (self.__bigFaceBound[1] - self.__bigFaceBound[0])/(noSlices - 1)
            for i in range (noSlices - 1):
                self.__bigFaceConst.append(self.__bigFaceConst[i] - const)
        
        if self.__faceType == 1: # Mặt lưng và bụng
            self.__smallFaceConst.append(self.__smallFaceBound[1])
            const = (self.__smallFaceBound[1] - self.__smallFaceBound[0])/(noSlices - 1)
            for i in range (noSlices - 1):
                self.__smallFaceConst.append(self.__smallFaceConst[i] - const)

    # Tính diện tích khiếm khuyết
    # output: 
    def defectArea(self, defect = np.zeros((300, 520), np.uint8)):
        # Nếu không truyền gì vào defect, thuật toán mặc định sử dụng __allDefects làm ảnh nguồn
        if defect.any() == 0:
            defect = self.__allDefects

        # Tính hệ số diện tích
        self.constK()

        # Khởi tạo biến lưu diện tích khiếm khuyết
        defArea = 0

        # Kiểm tra dạng mặt xoài
        if self.__faceType == 0: # Mặt cạnh
            typeK = self.__bigFaceConst
        if self.__faceType == 1: # Mặt lưng hoặc bụng
            typeK = self.__smallFaceConst

        # Tính diện tích khiếm khuyết
        for i in range (len(self.__slices)):
            idx = int((self.__intensity[i] - min(self.__intensity))/min(self.__intensity))
            defArea += np.count_nonzero(defect == self.__intensity[i])*typeK[idx]

        # Lưu tổng diện tích khiếm khuyết
        self.__defArea = round(defArea, 4)

        return self.__defArea

    # Hiển thị kết quả xác định và tính diện tích các vết khiếm khuyết trên bề mặt xoài
    def showResult(self):
        bound = self.__defectCont.copy()
        h, w = self.__face.shape[:2]
        if self.__dispType == 1:
            if len(self.__defects) > 0:
                for defect in self.__defects:
                    # Xác định vị trí hiển thị diện tích cho từng vết khiếm khuyết
                    _, whiteDef = cv2.threshold(defect, 1, 255, cv2.THRESH_BINARY)
                    M = cv2.moments(whiteDef, True)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Add text
                    cv2.putText(bound, str(round(self.defectArea(defect),2)), (cX - 10, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, self.__fontSize2, (255, 255, 255), 2, cv2.LINE_8)
            else:
                cv2.putText(bound, str(f"{round(self.defectArea(self.__allDefects),2)} mm2"), (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, self.__fontSize1, (255, 255, 255), 2, cv2.LINE_8)
        
        if self.__dispType == 0:
            cv2.putText(bound, str(f"{round(self.defectArea(self.__allDefects),2)} mm2"), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, self.__fontSize1, (255, 255, 255), 2, cv2.LINE_8)
        return bound