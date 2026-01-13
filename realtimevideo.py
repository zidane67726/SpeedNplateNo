import cv2
from ultralytics import YOLO
import easyocr

VIDEO_PATH = r"D:\Screen Recordings\Indian_Traffic_Footage_Pixels_1080P.mp4"

from ultralytics import YOLO

model_path = r"D:\best.pt"
model = YOLO(model_path)


# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(VIDEO_PATH)
last_plate = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

    results = model(frame, conf=0.3, iou=0.5)
    for r in results:
        for box in r.boxes:
            # Extract plate bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop just the plate region
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # Convert to grayscale for OCR
            gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            # Run EasyOCR on the cropped plate
            ocr_results = reader.readtext(gray_plate, detail=1)

            for _, text, prob in ocr_results:
                text = text.replace(" ", "").upper()
                if prob > 0.5 and len(text) >= 4:
                    if text != last_plate:
                        print("Detected Plate:", text)
                        last_plate = text

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

    cv2.imshow("Plate Detection + OCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
