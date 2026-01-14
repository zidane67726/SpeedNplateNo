import os
import cv2

# ===================== CONFIGURATION =====================
# ✅ Put FULL paths (so you always know where files are)
INPUT_VIDEO  = r"D:\Screen Recordings\Screen Recording 2026-01-13 164111.mp4"
OUTPUT_VIDEO = r"D:\Screen Recordings\enhanced_plate_video.mp4"

# Enhancement strength (safe defaults for ANPR)
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID  = (8, 8)

# Sharpening strength
SHARPEN_ALPHA = 1.4   # higher = sharper (keep 1.2–1.6)
BLUR_SIGMA    = 1.0   # blur used for unsharp masking

# FAST denoise (recommended on CPU)
USE_FAST_DENOISE = True

# =========================================================

def main():
    print("Input :", INPUT_VIDEO)
    print("Output:", OUTPUT_VIDEO)

    # 1) Check input exists
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"❌ Input video not found:\n{INPUT_VIDEO}")

    # 2) Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("❌ OpenCV could not open the input video (path/codec issue).")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # handle 0 or NaN
        fps = 30

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video opened ✅ | {w}x{h} | fps={fps:.2f} | frames={total}")

    # 3) Setup writer (try mp4v first; fallback to avi if needed)
    out_path = OUTPUT_VIDEO
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    if not out.isOpened():
        print("mp4v failed ❗ Trying XVID (.avi) fallback...")
        out_path = os.path.splitext(OUTPUT_VIDEO)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    if not out.isOpened():
        cap.release()
        raise RuntimeError("❌ VideoWriter could not be opened. Codec not supported on this system.")

    # 4) Enhancement tools
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)

    # 5) Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video.")
            break

        # --- FAST denoise (good for compressed screen recordings / CCTV) ---
        if USE_FAST_DENOISE:
            frame = cv2.bilateralFilter(frame, 7, 50, 50)
        else:
            # slower but sometimes cleaner
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

        # --- Contrast enhancement (CLAHE on L channel) ---
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        frame = cv2.merge((l, a, b))
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

        # --- Unsharp mask sharpening (improves plate character edges) ---
        blurred = cv2.GaussianBlur(frame, (0, 0), BLUR_SIGMA)
        frame = cv2.addWeighted(frame, SHARPEN_ALPHA, blurred, -(SHARPEN_ALPHA - 1.0), 0)

        out.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total} frames...")

    # 6) Cleanup
    cap.release()
    out.release()

    print("\n✅ DONE")
    print("Saved enhanced video at:", out_path)
    print("File exists now? ->", os.path.exists(out_path))


if __name__ == "__main__":
    main()

