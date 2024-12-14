from ultralytics import YOLO
import cv2
import cvzone
import time, math

# CCTV
# rtsp_source = "rtsp://admin:admin12345@192.168.6.2:554/cam/realmonitor?channel=2&subtype=0" # DAHUA
rtsp_source = "rtsp://admin:gcc12345@192.168.6.7:554/Streaming/Channels/101/" # HIKVISION
cap = cv2.VideoCapture(rtsp_source)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# Video
# cap = cv2.VideoCapture("../Videos/people.mp4")
# cap.set(3,640) # lebar {id = 3} dengan size 640
# cap.set(4,720) # tinggi {id = 4} dengan size 720

model = YOLO("../Yolo-Weights/yolov8n.pt")

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

# Inisialisasi waktu untuk perhitungan FPS
prev_time = 0

while True:
    try:
        success, img = cap.read()

        if not success:
            print("Gagal mendapatkan frame, mencoba lagi...")
            time.sleep(0.2)  # Tunggu 0.2 detik sebelum mencoba lagi
            continue

        # Menghitung waktu sekarang
        current_time = time.time()

        # Hitung FPS
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        img_resized = cv2.resize(img, (1020, 640)) # CCTV
        # img_resized = img # Videos

        # Tampilkan FPS di frame
        cv2.putText(img_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        results = model(img_resized, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1,y2-y1
                cvzone.cornerRect(img_resized,(x1,y1,w,h))

                # Menampilkan Tulisan Persentase
                conf = math.ceil((box.conf[0]*100))/100
                print(conf)

                # Class Name
                cls = int(box.cls[0])
                print()

                cvzone.putTextRect(img_resized, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)),scale=0.7,thickness=1)

        cv2.imshow("Image", img_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Tutup jendela dan stream
cap.release()
cv2.destroyAllWindows()