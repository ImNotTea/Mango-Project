import cv2
import numpy as np
import os, sys
import glob

sys.path.append(os.getcwd())

from Library.Mango import Mango
from Library.MangoFace import MangoFace
from Library import ProLib

# List video xoài
videoList = glob.glob("Input Source/Mango Video/Mango 26.11.22/New System/No Tape/*.avi")

# Kiểm tra đường dẫn và list video
if not len(videoList):
    sys.exit("No video in the list or wrong path \nCheck it out!")

# Chương trình chính
for videoPath in videoList:
    video = cv2.VideoCapture(videoPath)
 
    if not video.isOpened():
        sys.exit("Cannot read the video")

    # Tạo Object cho mỗi quả xoài
    confPath = "Library/Config/Mango_1.json" # Đường dẫn file config (chóng bóp méo ảnh).
    mango1 = Mango(confPath)

    # Thu thập và xử lý các khung hình
    while video.isOpened():
        ret, frame = video.read()
    
        if ret == True:
            # Đối với hệ thống mới, các khung hình xoài cần được xoay 90 độ trước khi xử lý.
            if mango1.need2Rotate():
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = ProLib.scale(frame, 0.5)

            # Tiền xử lý khung hình xoài
            tracked, inWorkingZone, roiCordinate = mango1.trackingMango(frame)
            roiFrame = ProLib.drawROI(frame, roiCordinate, inWorkingZone, 2)
            rmbgFrame, cropMask = mango1.rmBackground(inWorkingZone)
            
            # Xuất kết quả tiền xử lý ra màn hình
            cv2.imshow("Frame", roiFrame)
            cv2.imshow("Tracked Mango", tracked)
            cv2.imshow("Remove BG", rmbgFrame)

            # Xử lý nút nhấn
            key = cv2.waitKey(20)
            
            if key == ord('q'): # Skip video hiện tại
                print("Skip the video")
                cv2.destroyAllWindows()
                break

            if key == ord('s'): # Dừng chương trình
                print("Stop processing")
                cv2.destroyAllWindows()
                break
        else:
            print("Video run out")
            cv2.destroyAllWindows()
            break

    # 
    if key == ord('q'): 
        continue
    if key == ord('s'):
        break

    # Xác định 4 mặt xoài
    bigFaces, smallFaces = mango1.detectMainFaces()

    # Tạo 4 objects tương đương với 4 mặt xoài
    bigFace1 = MangoFace(bigFaces[0], 0, confPath)
    bigFace2 = MangoFace(bigFaces[1], 0, confPath)
    smallFace1 = MangoFace(smallFaces[0], 1, confPath)
    smallFace2 = MangoFace(smallFaces[1], 1, confPath)

    # Cắt lát bề mặt xoài
    bigFace1.slicing()
    bigFace2.slicing()
    smallFace1.slicing()
    smallFace2.slicing()

    # Xác định các vết khiếm khuyết bề mặt
    _, slicedDef1, defCont1 = bigFace1.findDefect()
    _, slicedDef2, defCont2 = bigFace2.findDefect()
    _, slicedDef3, defCont3 = smallFace1.findDefect()
    _, slicedDef4, defCont4 = smallFace2.findDefect()

    # Tính toán và hiển thị diện tích các vết khiếm khuyết
    res1 = bigFace1.showResult()
    res2 = bigFace2.showResult()
    res3 = smallFace1.showResult()
    res4 = smallFace2.showResult()

    # Hiển thị kết quả
    mainFaces = ProLib.stackImages((bigFaces[0], bigFaces[1], smallFaces[0], smallFaces[1]), 0)
    slicedDef = ProLib.stackImages((slicedDef1, slicedDef2, slicedDef3, slicedDef4), 0)
    defCont = ProLib.stackImages((defCont1, defCont2, defCont3, defCont4), 0)
    res = ProLib.stackImages((res1, res2, res3, res4), 0)

    # cv2.imshow("Sliced Defects", slicedDef)
    # cv2.imshow("Defect Contours", defCont)
    # cv2.imshow("Main Faces", mainFaces)
    cv2.imshow("Res", res)

    cv2.waitKey()
    cv2.destroyAllWindows()