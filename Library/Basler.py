import cv2
import numpy as np
from pypylon import pylon
import os, sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


sys.path.append(f"{os.getcwd()}/Library")

from ProLib import *

# Tìm hiểu thêm thông tin tại đây:
# https://www.pythonforthelab.com/blog/getting-started-with-basler-cameras/

# Hiển thị thông tin các camera đang kết nối với thiết bị
def getCamInfo():
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    for device in devices:
        print(device.GetFriendlyName())

# Hiệu chỉnh Camera
# Tham khảo tại: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# Xử lý ảnh bàn cờ để xuất các file config giúp hiệu chỉnh camera
# checkerboardFolderPath: Folder chứa ảnh bàn cờ vua
# savingPath: Đường dẫn lưu các file config
def calibCam(CheckerboardFolderPath, savingPath):
    # Xác định kích thước bàn cờ (width - 1, height - 1)
    CHECKERBOARD = (6, 9)

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Danh sách ảnh bàn cờ chụp bởi camera basler
    images = glob.glob(f'{CheckerboardFolderPath}/*.png')

    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            threedpoints.append(objectp3d)

            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)

    ret, matrix, distortion, rotateVecs, translateVecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    # Lưu mảng matrix và distortion
    mat = pd.DataFrame(matrix)
    dist = pd.DataFrame(distortion)
    mat.to_csv(f"{savingPath}/Matrix.csv", header=False, index=False)
    dist.to_csv(f"{savingPath}/Distort.csv", header=False, index=False)

    return matrix, distortion, rotateVecs, translateVecs

# Hiệu chỉnh hình ảnh (chống bóp méo hình ảnh)
# img: Ảnh gốc (bị bóp méo)
# matrixPath: Đường dẫn file matrix
# distortPath: Đường dẫn file distort
# output: Hình ảnh sau khi được chống bóp méo
def undistortImg(img, matrixPath, distortPath):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h))
    # Đọc các file config
    matrix = pd.read_csv(matrixPath, header=None).values
    distortion = pd.read_csv(distortPath, header=None).values
    # Chống bóp méo hình ảnh
    res = cv2.undistort(img, matrix, distortion, None, newcameramtx)
    
    res = res[roi[1]:roi[3], roi[0]:roi[2]]

    return res

# Truy cập camera liên tục - Ghi hình
# savingPath: Đường dẫn đến Folder lưu video sau khi ghi hình
# videoName: Đặt tên video theo format (name.avi)
def acquireContinuous(savingPath, videoName):
    # Tạo các objects
    tl_factory = pylon.TlFactory.GetInstance()
    camera = pylon.InstantCamera()
    camera.Attach(tl_factory.CreateFirstDevice())

    # Đường dẫn file config
    # File này được xuất ra từ app pylon Viewer sau khi được tinh chỉnh bằng tay
    # các thông số về độ sáng, lấy nét, ...
    nodeFile = "Library/Calib Cam/New System/acA1920-40uc_23065001.pfs"

    # Mở camera
    camera.Open()
    pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Chuyển đổi hệ màu sang RGB (Opencv)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Đường dẫn các file config thu được từ hàm undistortImg
    matPath = "Library/Calib Cam/New System/Matrix.csv"
    distPath = "Library/Calib Cam/New System/Distort.csv"

    # Save video
    # Đối số cuối cùng của hàm cv2.VideoWriter là kích thước khung hình
    # Kích thước này phải trùng khớp với kích thước khi save của khung hình
    # Nếu không, video sau khi lưu sẽ bị lỗi
    result = cv2.VideoWriter(f"{savingPath}/{videoName}", cv2.VideoWriter_fourcc(*'XVID'), 30, (780, 350))

    # Khởi tạo biến trạng thái
    recording = False

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
       
        if grab.GrabSucceeded():
            frame = converter.Convert(grab)
            frame = frame.GetArray()
            frame = scale(frame, 0.5)

            # Chống bóp méo hình ảnh
            frame = undistortImg(frame, matPath, distPath)

            # Hiệu chỉnh kích thước khung hình sao cho trùng khớp với khai báo ở line 140
            frame = frame[230:580, 220:1000, :]
            
            if recording:
                # Lưu khung hình
                result.write(frame)
            cv2.imshow("Frame", frame)

        # Sự kiện nút nhấn
        key = cv2.waitKey(1)
        # Thoát bằng nút 'q'
        if key == ord('q'):
            print("Stop acquiring camera")
            break

        # Bắt đầu ghi hình bằng nút 'r'
        # Sau khi ghi hình nhấn nhút 'q' để thoát
        if key == ord('r'):
            recording = True

    camera.Close()
    result.release()
    cv2.destroyAllWindows()

# Chụp ảnh
# savingPath: Đường dẫn đến Folder lưu ảnh sau khi chụp
# picName: Đặt tên ảnh theo format (name.png hoặc name.jpg)
# Nên lưu ảnh với định dạng png để có chất lượng tốt nhất
def captureImage(savingPath, picName):
    # Tạo các objects
    tl_factory = pylon.TlFactory.GetInstance()
    camera = pylon.InstantCamera()
    camera.Attach(tl_factory.CreateFirstDevice())

    # Đường dẫn file config
    # File này được xuất ra từ app pylon Viewer sau khi được tinh chỉnh bằng tay
    # các thông số về độ sáng, lấy nét, ...
    nodeFile = "Library/Calib Cam/New System/acA1920-40uc_23065001.pfs"

    # Mở camera
    camera.Open()
    pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Chuyển đổi hệ màu sang RGB (Opencv)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Đường dẫn các file config
    matPath = "Library/Calib Cam/New System/Matrix.csv"
    distPath = "Library/Calib Cam/New System/Distort.csv"

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
       
        if grab.GrabSucceeded():
            frame = converter.Convert(grab)
            frame = frame.GetArray()
            frame = scale(frame, 0.5)

            # Undistort frame
            frame = undistortImg(frame, matPath, distPath)

            # Hiển thị khung hính
            cv2.imshow("Frame", frame)

        # Sự kiện nút nhấn
        key = cv2.waitKey(1)
        # Thoát bằng nút 'q'
        if key == ord('q'):
            print("Stop acquiring camera")
            break

        # Chụp ảnh bằng nút 'c'
        if key == ord('c'):
            print("Captured")
            cv2.imwrite(f"{savingPath}/{picName}", frame)

    camera.Close()
    cv2.destroyAllWindows()