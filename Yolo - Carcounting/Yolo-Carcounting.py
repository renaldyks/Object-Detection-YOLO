import numpy as np
from sympy import limit
from sympy.physics.units import current
from ultralytics import YOLO
import cv2
import cvzone
import time, math, datetime
from sort import *

# RTSP Source
# rtsp_source = "rtsp://admin:gcc12345@192.168.6.7:554/Streaming/Channels/101/" # HIKVISION
# rtsp_source = "rtsp://admin:admin12345@192.168.6.2:554/cam/realmonitor?channel=3&subtype=0" # DAHUA
# rtsp_source = "rtsp://admin:admin12345@192.168.6.2:554/cam/realmonitor?channel=2&subtype=0" # DAHUA
# rtsp_source = "rtsp://admin:admin12345@192.168.6.2:554/cam/realmonitor?channel=4&subtype=0" # DAHUA

rtsp_source = "../Videos/cars.mp4" # VIDEO

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [300, 350, 673, 350]

# Inisialisasi model YOLO
model = YOLO("../Yolo-Weights/yolov8l.pt")

className = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Fungsi untuk mereset jumlah upaya koneksi ulang
def reset_attempts():
    return 50

# Fungsi untuk memproses video dan mengatur reconnect
def process_video(attempts):
    totalCount = []
    # Inisialisasi waktu untuk perhitungan FPS
    prev_time = time.time()

    while True:
        success, img = camera.read()

        if not success:
            print("Gagal mendapatkan frame, mencoba reconnect...")
            camera.release()

            if attempts > 0:
                time.sleep(5)
                return True
            else:
                return False

        # Menghitung waktu sekarang
        current_time = time.time()

        # Hitung FPS
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        img_resized = cv2.resize(img, (1300, 700))  # Resize frame

        # Tampilkan FPS di frame
        cv2.putText(img_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        img_region = cv2.bitwise_and(img_resized, mask)

        # Deteksi dengan YOLO
        results = model(img_region, stream=True)

        # Tracker
        detections = np.empty((0,5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Menampilkan Tulisan Persentase
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])

                currentClass = className[cls]

                if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                    # cvzone.putTextRect(img_resized, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=3)
                    # cvzone.cornerRect(img_resized, (x1, y1, w, h), l=5, rt=5)
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections,currentArray))

        # Tracker
        resultsTracker = tracker.update(detections)
        cv2.line(img_resized,(limits[0], limits[1]), (limits[2], limits[3]),(0, 0, 255), 5)

        for result in resultsTracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img_resized, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,255))
            cvzone.putTextRect(img_resized, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

            cx, cy = x1+w//2, y1+h//2
            cv2.circle(img_resized,(cx,cy),5,(255,0,255),cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1]+20:
                if totalCount.count(Id) == 0:
                    totalCount.append(Id)

        cvzone.putTextRect(img_resized,f'Count: {len(totalCount)}',(50, 50))

        # Tampilkan hasil frame
        cv2.imshow("Image", img_resized)
        # cv2.imshow("ImageRegion", img_region)

        # Tekan 'q' untuk keluar
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        cv2.waitKey(0)

    return False

# Inisialisasi variabel recall untuk reconnect dan attempts untuk jumlah upaya
recall = True
attempts = reset_attempts()

while recall:
    # Membuka stream RTSP
    camera = cv2.VideoCapture(rtsp_source)

    if camera.isOpened():
        print("[INFO] Camera connected at " +
              datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"))
        attempts = reset_attempts()  # Reset attempts saat berhasil terkoneksi
        recall = process_video(attempts)  # Proses video
    else:
        print("Camera not opened " +
              datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"))
        camera.release()
        attempts -= 1
        print(f"Remaining attempts: {attempts}")

        # Beri jeda untuk mencoba kembali
        time.sleep(5)
        continue

# Tutup jendela dan stream
cv2.destroyAllWindows()
