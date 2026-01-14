import cv2
import numpy as np
import os
import time
import csv
import re
from ultralytics import YOLO
import easyocr
from collections import defaultdict, deque

# ===================== CONFIGURATION =====================
VIDEO_PATH = r"D:\Screen Recordings\Indian_Traffic_Footage_Pixels_1080P.mp4"
PLATE_MODEL_PATH = r"D:\best.pt"

OUTPUT_FOLDER = "violations"
CSV_FILE = "violations.csv"

# Line positions (adjust to match your video)
START_LINE = ((10, 650), (3000, 600))
END_LINE   = ((10, 400), (3180, 450))

DISTANCE_BETWEEN_LINES = 12.0  # meters
SPEED_LIMIT = 40               # km/h

MIN_TIME_THRESHOLD = 0.3
MAX_TIME_THRESHOLD = 10.0

# Detection / OCR settings
VEHICLE_CONF = 0.40
VEHICLE_IOU  = 0.50
PLATE_DET_CONF = 0.15  # lower helps find tiny plates; we'll filter later

OCR_ACCEPT_CONF = 0.40
MIN_PLATE_LENGTH = 4

# Temporal stabilization (POC stability)
PLATE_HISTORY_SIZE = 12          # store multiple readings
PLATE_BUFFER_FRAMES = 45         # keep best plate crops over ~1.5s at 30fps
MAX_ACTIVE_VEHICLES = 200        # safety

# Display
DISPLAY_SCALE = 0.8
WINDOW_NAME = "Speed & Plate Detection"

# =========================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["VehicleID", "Plate", "Speed_kmh", "Confidence", "Timestamp", "ImagePath"])

print("Loading models...")
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO(PLATE_MODEL_PATH)
ocr_reader = easyocr.Reader(["en"], gpu=False)
print("✓ Models loaded successfully!")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video {VIDEO_PATH}")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps != fps:
    fps = 30.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f"Video: {w}x{h} @ {fps:.2f} FPS, {total_frames} frames")

processed_violations = set()
frame_count = 0

# Vehicle memory (per ByteTrack ID)
vehicle_memory = {}  # vid -> dict


# ===================== HELPER FUNCTIONS =====================

def point_line_distance(px, py, line):
    """Signed distance from point to line; sign change = crossing."""
    (x1, y1), (x2, y2) = line
    return ((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1))

def clean_plate_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def is_valid_plate(text):
    if len(text) < MIN_PLATE_LENGTH:
        return False
    has_letter = any(c.isalpha() for c in text)
    has_number = any(c.isdigit() for c in text)
    return has_letter and has_number

def sharpness_score(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def enhance_plate_for_ocr(plate_bgr):
    """
    Return a list of processed images (grayscale/binary) for OCR attempts.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return []

    # Upscale for tiny plates
    up = cv2.resize(plate_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # Fast denoise
    den = cv2.bilateralFilter(gray, 11, 17, 17)

    # Adaptive threshold
    th1 = cv2.adaptiveThreshold(
        den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Otsu
    _, th2 = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morph close to connect characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)

    # Histogram equalization
    eq = cv2.equalizeHist(gray)

    return [th1, th2, morph, eq, gray]

def extract_plate_text(plate_bgr):
    """
    OCR on multiple processed versions; return best valid (text, conf).
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return None, 0.0

    best_text, best_conf = None, 0.0
    try:
        processed = enhance_plate_for_ocr(plate_bgr)
        for img in processed:
            results = ocr_reader.readtext(img, detail=1, paragraph=False)
            for _, text, conf in results:
                t = clean_plate_text(text)
                if conf > best_conf and conf >= OCR_ACCEPT_CONF and is_valid_plate(t):
                    best_text, best_conf = t, float(conf)
    except Exception:
        pass

    return best_text, best_conf

def get_consensus_plate(plate_history):
    """
    plate_history: list[(text, conf)]
    Choose by (count * avg_conf)
    """
    if not plate_history:
        return None, 0.0

    scores = defaultdict(lambda: {"count": 0, "total_conf": 0.0})
    for t, c in plate_history:
        scores[t]["count"] += 1
        scores[t]["total_conf"] += float(c)

    best_t, best_score, best_avg = None, 0.0, 0.0
    for t, d in scores.items():
        avg = d["total_conf"] / max(d["count"], 1)
        score = d["count"] * avg
        if score > best_score:
            best_score = score
            best_t = t
            best_avg = avg

    return best_t, best_avg

def make_plate_preview_enhanced(plate_bgr):
    """
    Nice-looking preview crop (BGR), not necessarily same as OCR binary.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return None

    img = cv2.resize(plate_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img = cv2.bilateralFilter(img, 7, 50, 50)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img

def draw_reference_lines(frame, scale=1.0):
    start_scaled = (
        (int(START_LINE[0][0] * scale), int(START_LINE[0][1] * scale)),
        (int(START_LINE[1][0] * scale), int(START_LINE[1][1] * scale))
    )
    end_scaled = (
        (int(END_LINE[0][0] * scale), int(END_LINE[0][1] * scale)),
        (int(END_LINE[1][0] * scale), int(END_LINE[1][1] * scale))
    )

    cv2.line(frame, start_scaled[0], start_scaled[1], (0, 100, 0), 6)
    cv2.line(frame, start_scaled[0], start_scaled[1], (0, 255, 0), 3)
    cv2.putText(frame, "START", (start_scaled[0][0] + 10, start_scaled[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.line(frame, end_scaled[0], end_scaled[1], (0, 0, 100), 6)
    cv2.line(frame, end_scaled[0], end_scaled[1], (0, 0, 255), 3)
    cv2.putText(frame, "END", (end_scaled[0][0] + 10, end_scaled[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def draw_status_bar(frame, active_vehicles, violations, frame_num, total):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    progress = (frame_num / total) * 100 if total > 0 else 0
    cv2.putText(frame, f"Frame: {frame_num}/{total} ({progress:.1f}%)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Active: {active_vehicles}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Violations: {violations}", (200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def draw_plate_preview(frame_disp, vehicle):
    """
    Shows the ACTUAL detected plate crop (enhanced) on top-right.
    This makes the POC look stable and honest.
    """
    preview = vehicle.get("last_plate_crop_enh", None)
    text = vehicle.get("last_plate_text", None)
    conf = vehicle.get("last_plate_conf", 0.0)

    if preview is None or preview.size == 0:
        return

    target_w, target_h = 340, 130
    pv = cv2.resize(preview, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    x2 = frame_disp.shape[1] - 10
    y1 = 75
    x1 = x2 - target_w
    y2 = y1 + target_h

    # bg
    cv2.rectangle(frame_disp, (x1 - 6, y1 - 32), (x2 + 6, y2 + 6), (0, 0, 0), -1)
    cv2.rectangle(frame_disp, (x1 - 6, y1 - 32), (x2 + 6, y2 + 6), (255, 255, 255), 2)

    frame_disp[y1:y2, x1:x2] = pv

    label = "PLATE: "
    if text:
        label += f"{text} ({conf:.2f})"
        color = (0, 255, 0)
    else:
        label += "READING..."
        color = (0, 165, 255)

    cv2.putText(frame_disp, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def get_best_plate_from_buffer(vehicle):
    """
    vehicle["plate_buffer"] contains tuples (score, plate_crop_bgr).
    We OCR the top few and vote.
    """
    buf = vehicle.get("plate_buffer", None)
    if not buf:
        return None, 0.0

    # take top-N crops by score
    items = sorted(list(buf), key=lambda x: x[0], reverse=True)[:10]

    reads = []
    for _, crop in items:
        t, c = extract_plate_text(crop)
        if t:
            reads.append((t, c))

    if reads:
        # Add to history & consensus
        for t, c in reads:
            if len(vehicle["plate_history"]) < PLATE_HISTORY_SIZE:
                vehicle["plate_history"].append((t, c))
        return get_consensus_plate(vehicle["plate_history"])

    return None, 0.0


# ===================== MAIN LOOP =====================

print("\nProcessing video... Press 'Q' or ESC to quit")
print("=" * 60)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # occasional progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames else 0
            print(f"Progress: {progress:.1f}% | Active: {len(vehicle_memory)} | Violations: {len(processed_violations)}")

        # Vehicle tracking
        results = vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=VEHICLE_CONF,
            iou=VEHICLE_IOU,
            classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
            verbose=False
        )

        frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        draw_reference_lines(frame_disp, DISPLAY_SCALE)

        # Cleanup memory if too big (safety)
        if len(vehicle_memory) > MAX_ACTIVE_VEHICLES:
            # remove oldest keys roughly
            for k in list(vehicle_memory.keys())[:len(vehicle_memory) - MAX_ACTIVE_VEHICLES]:
                vehicle_memory.pop(k, None)

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                vid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # bottom-center point for crossing
                cx = (x1 + x2) // 2
                cy = y2

                if vid not in vehicle_memory:
                    vehicle_memory[vid] = {
                        "prev_dist_start": None,
                        "prev_dist_end": None,
                        "line1_frame": None,
                        "line2_frame": None,
                        "speed": None,
                        "violation_recorded": False,

                        # Plate stabilization
                        "plate_history": [],
                        "plate_buffer": deque(maxlen=PLATE_BUFFER_FRAMES),

                        # For on-screen preview (actual plate crop)
                        "last_plate_crop": None,
                        "last_plate_crop_enh": None,
                        "last_plate_text": None,
                        "last_plate_conf": 0.0
                    }

                vehicle = vehicle_memory[vid]

                # ===================== PLATE DETECTION INSIDE VEHICLE =====================
                # Always try plate detection while tracking (POC stability)
                if not vehicle["violation_recorded"]:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size > 0:
                        try:
                            pr = plate_model(vehicle_crop, conf=PLATE_DET_CONF, verbose=False)[0]
                            if pr.boxes is not None and len(pr.boxes) > 0:
                                # choose best plate box by confidence * area
                                best = None
                                best_score = -1
                                for pb in pr.boxes:
                                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                                    pc = float(pb.conf.item()) if pb.conf is not None else 0.0
                                    px1, py1 = max(0, px1), max(0, py1)
                                    px2, py2 = min(vehicle_crop.shape[1], px2), min(vehicle_crop.shape[0], py2)
                                    area = max(0, px2 - px1) * max(0, py2 - py1)
                                    score = pc * area
                                    if score > best_score:
                                        best_score = score
                                        best = (px1, py1, px2, py2, pc)

                                if best is not None:
                                    px1, py1, px2, py2, pc = best
                                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                                    if plate_crop.size > 0:
                                        # Store latest crop for preview
                                        vehicle["last_plate_crop"] = plate_crop.copy()
                                        enh_preview = make_plate_preview_enhanced(plate_crop)
                                        vehicle["last_plate_crop_enh"] = enh_preview

                                        # Add to buffer with sharpness+size score
                                        sh = sharpness_score(plate_crop)
                                        area = (px2 - px1) * (py2 - py1)
                                        buf_score = sh * max(area, 1)
                                        vehicle["plate_buffer"].append((buf_score, plate_crop.copy()))

                                        # Try OCR quickly on this crop (may fail; preview still shows)
                                        t, c = extract_plate_text(plate_crop)
                                        if t:
                                            vehicle["last_plate_text"] = t
                                            vehicle["last_plate_conf"] = c
                                            if len(vehicle["plate_history"]) < PLATE_HISTORY_SIZE:
                                                vehicle["plate_history"].append((t, c))
                        except Exception:
                            pass

                # ===================== LINE CROSSING / SPEED =====================
                curr_start = point_line_distance(cx, cy, START_LINE)
                curr_end   = point_line_distance(cx, cy, END_LINE)

                # Start crossing
                if vehicle["prev_dist_start"] is not None and vehicle["line1_frame"] is None:
                    if vehicle["prev_dist_start"] * curr_start < 0:
                        vehicle["line1_frame"] = frame_count
                        print(f"  Vehicle {vid} crossed START (frame {frame_count})")

                # End crossing
                if vehicle["prev_dist_end"] is not None:
                    if vehicle["line1_frame"] is not None and vehicle["line2_frame"] is None:
                        if vehicle["prev_dist_end"] * curr_end < 0:
                            vehicle["line2_frame"] = frame_count
                            time_taken = (vehicle["line2_frame"] - vehicle["line1_frame"]) / fps

                            if MIN_TIME_THRESHOLD <= time_taken <= MAX_TIME_THRESHOLD:
                                speed_kmh = (DISTANCE_BETWEEN_LINES / time_taken) * 3.6
                                vehicle["speed"] = speed_kmh

                                print(f"  Vehicle {vid} crossed END (frame {frame_count})")
                                print(f"    → Speed: {speed_kmh:.2f} km/h (time: {time_taken:.2f}s)")

                                # ===================== VIOLATION =====================
                                if speed_kmh > SPEED_LIMIT and vid not in processed_violations:
                                    processed_violations.add(vid)
                                    vehicle["violation_recorded"] = True

                                    # Best plate: first consensus from history, else try best-from-buffer
                                    best_plate, plate_conf = get_consensus_plate(vehicle["plate_history"])
                                    if not best_plate:
                                        best_plate, plate_conf = get_best_plate_from_buffer(vehicle)

                                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                                    plate_display = best_plate if best_plate else "UNKNOWN"
                                    img_name = f"V{vid:04d}_{plate_display}_{int(speed_kmh)}kmh_{timestamp}.jpg"
                                    img_path = os.path.join(OUTPUT_FOLDER, img_name)

                                    evidence = frame[y1:y2, x1:x2].copy()
                                    if evidence.size > 0:
                                        cv2.rectangle(evidence, (0, 0), (evidence.shape[1], 110), (0, 0, 0), -1)
                                        cv2.rectangle(evidence, (0, 0), (evidence.shape[1], 110), (0, 0, 255), 2)

                                        cv2.putText(evidence, f"SPEED: {speed_kmh:.1f} km/h", (10, 32),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                        cv2.putText(evidence, f"LIMIT: {SPEED_LIMIT} km/h", (10, 62),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                        if best_plate:
                                            cv2.putText(evidence, f"PLATE: {best_plate} ({plate_conf:.2f})", (10, 95),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                                        cv2.imwrite(img_path, evidence)

                                    with open(CSV_FILE, "a", newline="") as f:
                                        writer = csv.writer(f)
                                        writer.writerow([
                                            vid,
                                            plate_display,
                                            round(speed_kmh, 2),
                                            f"{plate_conf:.2f}" if best_plate else "N/A",
                                            timestamp,
                                            img_path
                                        ])

                                    print(f"  ✓ VIOLATION: V{vid} | {plate_display} | {speed_kmh:.2f} km/h")
                            else:
                                print(f"  × Vehicle {vid}: Invalid time {time_taken:.2f}s")

                vehicle["prev_dist_start"] = curr_start
                vehicle["prev_dist_end"] = curr_end

                # ===================== VISUALIZATION =====================
                # Choose color
                if vehicle["speed"] is not None:
                    if vehicle["speed"] > SPEED_LIMIT:
                        color = (0, 0, 255)  # violation
                        thickness = 3
                    else:
                        color = (0, 255, 0)
                        thickness = 2
                elif vehicle["line1_frame"] is not None:
                    color = (0, 165, 255)  # measuring
                    thickness = 2
                else:
                    color = (255, 255, 0)  # tracking
                    thickness = 2

                x1_d, y1_d = int(x1 * DISPLAY_SCALE), int(y1 * DISPLAY_SCALE)
                x2_d, y2_d = int(x2 * DISPLAY_SCALE), int(y2 * DISPLAY_SCALE)
                cv2.rectangle(frame_disp, (x1_d, y1_d), (x2_d, y2_d), color, thickness)

                label = f"ID:{vid}"
                if vehicle["speed"] is not None:
                    label += f" | {vehicle['speed']:.1f}km/h"
                elif vehicle["line1_frame"] is not None:
                    label += " | Measuring..."

                # Show best current plate text (consensus)
                cons_plate, _ = get_consensus_plate(vehicle["plate_history"])
                if cons_plate:
                    label += f" | {cons_plate}"

                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_disp, (x1_d, y1_d - lh - 10), (x1_d + lw, y1_d), color, -1)
                cv2.putText(frame_disp, label, (x1_d, y1_d - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # IMPORTANT: Show ACTUAL plate crop preview (top-right)
                draw_plate_preview(frame_disp, vehicle)

        draw_status_bar(frame_disp, len(vehicle_memory), len(processed_violations), frame_count, total_frames)
        cv2.imshow(WINDOW_NAME, frame_disp)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q") or key == 27:
            print("\n✓ Stopped by user")
            break

except KeyboardInterrupt:
    print("\n✓ Interrupted by user")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()

# ===================== SUMMARY REPORT =====================
print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

try:
    with open(CSV_FILE, "r") as f:
        lines = f.readlines()

    if len(lines) <= 1:
        print("No speeding violations detected.")
    else:
        violations = len(lines) - 1
        print(f"Total violations: {violations}")
        print(f"Evidence folder: {os.path.abspath(OUTPUT_FOLDER)}")
        print(f"CSV report: {os.path.abspath(CSV_FILE)}")

        print("\n" + "-" * 60)
        print("Violation Details:")
        print("-" * 60)

        for i, line in enumerate(lines[1:], 1):
            parts = line.strip().split(",")
            if len(parts) >= 3:
                vid = parts[0]
                plate = parts[1]
                speed = parts[2]
                print(f"  {i}. Vehicle {vid:>4} | Plate: {plate:>12} | Speed: {speed:>6} km/h")

        print("-" * 60)

except Exception as e:
    print(f"Error reading results: {e}")

print("=" * 60)
