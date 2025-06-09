import cv2
import pytesseract
import torch

# Optional: set Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model
# model = torch.hub.load('yolov5', 'custom',
                    #    path=r'D:\projects\LPR\runs\train\exp2\weights\best.pt')
yolov5_path = r'D:\projects\LPR\yolov5' # Or use an absolute path: r'D:\projects\LPR\yolov5'

# Path to your custom weights
weights_path = r'D:\projects\LPR\runs\train\exp2\weights\best.pt'

# Load the model from the local source
model = torch.hub.load(
    yolov5_path,    # 1. Point to your local yolov5 repo
    'custom',
    path=weights_path,
    source='local'  # 2. THIS IS THE CRUCIAL PART
)


# Preprocessing function
def preprocess_plate(plate_img):
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("🎥 Press ENTER to capture OCR | Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Show live frame
    cv2.imshow("Live OCR Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # Enter key pressed
        print("🔍 Capturing frame and running OCR...")

        # Run YOLO detection on the current frame
        results = model(frame)
        detections = results.xyxy[0]

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # Preprocess and OCR
            processed = preprocess_plate(plate)
            text = pytesseract.image_to_string(processed, config='--psm 7').strip()

            # Display and print
            print(f"📄 OCR Result: {text}")
            cv2.imshow("Detected Plate", processed)
            break  # Only process one detection per Enter press

    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
