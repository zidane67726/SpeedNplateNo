import cv2
import numpy as np
import os
import time
import csv
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict

# ===================== CONFIGURATION =====================
VIDEO_PATH = r"D:\Screen Recordings\enhanced_plate_video.mp4" 
PLATE_MODEL_PATH = r"D:\best.pt"
OUTPUT_FOLDER = "violations"
CSV_FILE = "violations.csv"

# Line positions (adjust to match your video)
START_LINE = ((10, 650), (3000, 600))
END_LINE = ((10, 400), (3180, 450))

DISTANCE_BETWEEN_LINES = 12.0  # meters (real-world measured)
SPEED_LIMIT = 40  # km/h

MIN_TIME_THRESHOLD = 0.3
MAX_TIME_THRESHOLD = 10.0

# Enhanced parameters
PLATE_CONFIDENCE_THRESHOLD = 0.4
MIN_PLATE_LENGTH = 4
PLATE_HISTORY_SIZE = 5  # Collect multiple plate readings
DISPLAY_SCALE = 0.8  # Scale factor for display (adjust if window is too large)

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

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f"Video: {w}x{h} @ {fps:.2f} FPS, {total_frames} frames")

vehicle_memory = {}
processed_violations = set()
frame_count = 0

# ===================== HELPER FUNCTIONS =====================

def point_line_distance(px, py, line):
    """
    Calculate signed distance from point to line
    Sign change indicates line crossing
    """
    (x1, y1), (x2, y2) = line
    return ((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1))

def clean_plate_text(text):
    """Remove special characters, keep alphanumeric only"""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_valid_plate(text):
    """Validate plate format: must have both letters and numbers"""
    if len(text) < MIN_PLATE_LENGTH:
        return False
    has_letter = any(c.isalpha() for c in text)
    has_number = any(c.isdigit() for c in text)
    return has_letter and has_number

def enhance_plate_image(plate_img):
    """
    Apply multiple preprocessing techniques for better OCR
    Returns list of processed images to try
    """
    if plate_img is None or plate_img.size == 0:
        return []
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Bilateral filter + adaptive threshold
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh1 = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 2: Simple threshold
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 3: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    
    # Method 4: Histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    return [thresh1, thresh2, morph, equalized, gray]

def extract_plate_text(plate_img):
    """
    Enhanced OCR with multiple preprocessing methods
    Returns (text, confidence) tuple
    """
    if plate_img is None or plate_img.size == 0:
        return None, 0
    
    try:
        processed_images = enhance_plate_image(plate_img)
        all_results = []
        
        for img in processed_images:
            if img is None:
                continue
            
            ocr_results = ocr_reader.readtext(img, detail=1, paragraph=False)
            
            for bbox, text, conf in ocr_results:
                cleaned = clean_plate_text(text)
                
                if conf > PLATE_CONFIDENCE_THRESHOLD and is_valid_plate(cleaned):
                    all_results.append((cleaned, conf))
        
        # Return best result by confidence
        if all_results:
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[0]
        
        return None, 0
    
    except Exception as e:
        return None, 0

def get_consensus_plate(plate_history):
    """
    Get most reliable plate from multiple readings
    Uses frequency and confidence scoring
    """
    if not plate_history:
        return None, 0
    
    plate_scores = defaultdict(lambda: {'count': 0, 'total_conf': 0})
    
    for text, conf in plate_history:
        plate_scores[text]['count'] += 1
        plate_scores[text]['total_conf'] += conf
    
    # Find best plate: frequency × average confidence
    best_plate = None
    best_score = 0
    
    for text, data in plate_scores.items():
        avg_conf = data['total_conf'] / data['count']
        score = data['count'] * avg_conf
        
        if score > best_score:
            best_score = score
            best_plate = (text, avg_conf)
    
    return best_plate if best_plate else (None, 0)

def draw_status_bar(frame, active_vehicles, violations, frame_num, total):
    """Draw information overlay"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    progress = (frame_num / total) * 100 if total > 0 else 0
    
    # Status text
    status = f"Frame: {frame_num}/{total} ({progress:.1f}%)"
    vehicles = f"Active: {active_vehicles}"
    viols = f"Violations: {violations}"
    
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, vehicles, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, viols, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def draw_reference_lines(frame, scale=1.0):
    """Draw start and end lines with labels"""
    start_scaled = (
        (int(START_LINE[0][0] * scale), int(START_LINE[0][1] * scale)),
        (int(START_LINE[1][0] * scale), int(START_LINE[1][1] * scale))
    )
    end_scaled = (
        (int(END_LINE[0][0] * scale), int(END_LINE[0][1] * scale)),
        (int(END_LINE[1][0] * scale), int(END_LINE[1][1] * scale))
    )
    
    # Draw lines with glow effect
    cv2.line(frame, start_scaled[0], start_scaled[1], (0, 100, 0), 6)
    cv2.line(frame, start_scaled[0], start_scaled[1], (0, 255, 0), 3)
    cv2.putText(frame, "START", (start_scaled[0][0] + 10, start_scaled[0][1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.line(frame, end_scaled[0], end_scaled[1], (0, 0, 100), 6)
    cv2.line(frame, end_scaled[0], end_scaled[1], (0, 0, 255), 3)
    cv2.putText(frame, "END", (end_scaled[0][0] + 10, end_scaled[0][1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# ===================== MAIN PROCESSING LOOP =====================

print("\nProcessing video... Press 'Q' or ESC to quit")
print("=" * 60)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Active: {len(vehicle_memory)} | Violations: {len(processed_violations)}")
        
        # Vehicle detection with tracking (ByteTrack is more stable than manual tracking)
        results = vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            iou=0.5,
            classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
            verbose=False
        )
        
        # Create display frame
        frame_disp = frame.copy()
        frame_disp = cv2.resize(frame_disp, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        
        # Draw reference lines
        draw_reference_lines(frame_disp, DISPLAY_SCALE)
        
        # Process detections
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                vid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = y2  # Use bottom center for road-based tracking
                
                # Initialize vehicle memory
                if vid not in vehicle_memory:
                    vehicle_memory[vid] = {
                        "prev_dist_start": None,
                        "prev_dist_end": None,
                        "line1_frame": None,
                        "line2_frame": None,
                        "speed": None,
                        "plate_history": [],
                        "violation_recorded": False
                    }
                
                vehicle = vehicle_memory[vid]
                
                # ========== CONTINUOUS PLATE DETECTION ==========
                # Capture plates throughout tracking for better accuracy
                if len(vehicle["plate_history"]) < PLATE_HISTORY_SIZE and not vehicle["violation_recorded"]:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    
                    if vehicle_crop.size > 0:
                        try:
                            plate_results = plate_model(vehicle_crop, conf=0.25, verbose=False)
                            
                            for pr in plate_results:
                                if len(pr.boxes) > 0:
                                    pb = pr.boxes[0]  # Take best detection
                                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                                    
                                    if plate_crop.size > 0:
                                        text, conf = extract_plate_text(plate_crop)
                                        
                                        if text and conf > PLATE_CONFIDENCE_THRESHOLD:
                                            vehicle["plate_history"].append((text, conf))
                                            break
                        except Exception:
                            pass
                
                # ========== LINE CROSSING DETECTION ==========
                curr_start = point_line_distance(cx, cy, START_LINE)
                curr_end = point_line_distance(cx, cy, END_LINE)
                
                # Start line crossing
                if vehicle["prev_dist_start"] is not None and vehicle["line1_frame"] is None:
                    if vehicle["prev_dist_start"] * curr_start < 0:  # Sign change = crossing
                        vehicle["line1_frame"] = frame_count
                        print(f"  Vehicle {vid} crossed START line (frame {frame_count})")
                
                # End line crossing
                if vehicle["prev_dist_end"] is not None:
                    if vehicle["line1_frame"] is not None and vehicle["line2_frame"] is None:
                        if vehicle["prev_dist_end"] * curr_end < 0:  # Sign change = crossing
                            vehicle["line2_frame"] = frame_count
                            
                            time_taken = (vehicle["line2_frame"] - vehicle["line1_frame"]) / fps
                            
                            if MIN_TIME_THRESHOLD <= time_taken <= MAX_TIME_THRESHOLD:
                                speed_mps = DISTANCE_BETWEEN_LINES / time_taken
                                speed_kmh = speed_mps * 3.6
                                vehicle["speed"] = speed_kmh
                                
                                print(f"  Vehicle {vid} crossed END line (frame {frame_count})")
                                print(f"    → Speed: {speed_kmh:.2f} km/h (time: {time_taken:.2f}s)")
                                
                                # ========== VIOLATION PROCESSING ==========
                                if speed_kmh > SPEED_LIMIT and vid not in processed_violations:
                                    processed_violations.add(vid)
                                    vehicle["violation_recorded"] = True
                                    
                                    # Get best plate from history
                                    best_plate, plate_conf = get_consensus_plate(vehicle["plate_history"])
                                    
                                    # Final plate detection attempt if none found
                                    if not best_plate:
                                        vehicle_crop = frame[y1:y2, x1:x2]
                                        if vehicle_crop.size > 0:
                                            plate_results = plate_model(vehicle_crop, conf=0.2, verbose=False)
                                            
                                            for pr in plate_results:
                                                if len(pr.boxes) > 0:
                                                    pb = pr.boxes[0]
                                                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                                                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                                                    
                                                    if plate_crop.size > 0:
                                                        text, conf = extract_plate_text(plate_crop)
                                                        if text:
                                                            best_plate = text
                                                            plate_conf = conf
                                                            break
                                    
                                    # Save violation evidence
                                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                                    plate_display = best_plate if best_plate else "UNKNOWN"
                                    img_name = f"V{vid:04d}_{plate_display}_{int(speed_kmh)}kmh_{timestamp}.jpg"
                                    img_path = os.path.join(OUTPUT_FOLDER, img_name)
                                    
                                    # Create annotated evidence image
                                    vehicle_crop = frame[y1:y2, x1:x2].copy()
                                    
                                    # Add annotations
                                    cv2.rectangle(vehicle_crop, (0, 0), (vehicle_crop.shape[1], 100), (0, 0, 0), -1)
                                    cv2.rectangle(vehicle_crop, (0, 0), (vehicle_crop.shape[1], 100), (0, 0, 255), 2)
                                    
                                    cv2.putText(vehicle_crop, f"SPEED: {speed_kmh:.1f} km/h", (10, 30),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    cv2.putText(vehicle_crop, f"LIMIT: {SPEED_LIMIT} km/h", (10, 60),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    
                                    if best_plate:
                                        cv2.putText(vehicle_crop, f"PLATE: {best_plate}", (10, 90),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    cv2.imwrite(img_path, vehicle_crop)
                                    
                                    # Write to CSV
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
                
                # Update tracking distances
                vehicle["prev_dist_start"] = curr_start
                vehicle["prev_dist_end"] = curr_end
                
                # ========== VISUALIZATION ==========
                x1_d, y1_d = int(x1 * DISPLAY_SCALE), int(y1 * DISPLAY_SCALE)
                x2_d, y2_d = int(x2 * DISPLAY_SCALE), int(y2 * DISPLAY_SCALE)
                
                # Color coding based on status
                if vehicle["speed"] is not None:
                    if vehicle["speed"] > SPEED_LIMIT:
                        color = (0, 0, 255)  # Red: violation
                        thickness = 3
                    else:
                        color = (0, 255, 0)  # Green: safe speed
                        thickness = 2
                elif vehicle["line1_frame"] is not None:
                    color = (0, 165, 255)  # Orange: measuring
                    thickness = 2
                else:
                    color = (255, 255, 0)  # Cyan: tracking
                    thickness = 2
                
                # Draw vehicle box
                cv2.rectangle(frame_disp, (x1_d, y1_d), (x2_d, y2_d), color, thickness)
                
                # Create label
                label = f"ID:{vid}"
                
                if vehicle["speed"] is not None:
                    label += f" | {vehicle['speed']:.1f}km/h"
                elif vehicle["line1_frame"] is not None:
                    label += " | Measuring..."
                
                # Show plate if available
                best_plate, _ = get_consensus_plate(vehicle["plate_history"])
                if best_plate:
                    label += f" | {best_plate}"
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_disp, (x1_d, y1_d - label_h - 10), (x1_d + label_w, y1_d), color, -1)
                cv2.putText(frame_disp, label, (x1_d, y1_d - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw status bar
        draw_status_bar(frame_disp, len(vehicle_memory), len(processed_violations), frame_count, total_frames)
        
        # Display frame
        cv2.imshow("Speed & Plate Detection", frame_disp)
        
        # Key handling
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
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
    with open(CSV_FILE, 'r') as f:
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
            parts = line.strip().split(',')
            if len(parts) >= 3:
                vid = parts[0]
                plate = parts[1]
                speed = parts[2]
                print(f"  {i}. Vehicle {vid:>4} | Plate: {plate:>12} | Speed: {speed:>6} km/h")
        
        print("-" * 60)

except Exception as e:
    print(f"Error reading results: {e}")

print("=" * 60)