from ultralytics import YOLO
import cv2
import cvzone
import time, math, datetime

# RTSP Source
rtsp_source = "rtsp://admin:gcc12345@192.168.6.7:554/Streaming/Channels/101/" # HIKVISION
# rtsp_source = "rtsp://admin:admin12345@192.168.6.2:554/cam/realmonitor?channel=3&subtype=0" # DAHUA
# rtsp_source = "rtsp://admin:admin12345@192.168.6.2:554/cam/realmonitor?channel=2&subtype=0" # DAHUA
# rtsp_source = "rtsp://admin:admin12345@192.168.6.2:554/cam/realmonitor?channel=4&subtype=0" # DAHUA

# Inisialisasi model YOLO
model = YOLO("../Yolo-Weights/yolov8x.pt")

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

        img_resized = cv2.resize(img, (1020, 640))  # Resize frame

        # Tampilkan FPS di frame
        cv2.putText(img_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Deteksi dengan YOLO
        results = model(img_resized, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img_resized, (x1, y1, w, h))

                # Menampilkan Tulisan Persentase
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                cvzone.putTextRect(img_resized, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

        # Tampilkan hasil frame
        cv2.imshow("Image", img_resized)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
