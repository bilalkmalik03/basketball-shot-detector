# save as test_video_shot_detector.py
import cv2
from ultralytics import YOLO
from pipeline import ShotDetector
import os

ROOT = os.path.dirname(os.path.dirname(__file__))  # goes up from src/ to project root
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

VIDEO_IN = r"C:\Users\bilal\Videos\Screen Recordings\hooptest.mp4"

VIDEO_OUT = "outputs/bballtest_annotated.mp4"
FPS = 30.0   # set to actual video fps if known

# Load your model(s)
# Option A: one custom model that detects both classes
model = YOLO(os.path.join(ROOT, "models", "basketball_hoop", "weights", "best.pt"))

BALL_NAMES = {"sports ball", "basketball"}  # adapt to your class names
HOOP_NAMES = {"hoop", "basketball_hoop", "rim"}

def pick(objects, wanted_names):
    # Pick highest-conf object matching any name in wanted_names
    best = None
    for b in objects:
        cls_name = model.names[int(b.cls)]
        if cls_name in wanted_names:
            if (best is None) or (b.conf > best.conf):
                best = b
    return best

sd = ShotDetector(fps=FPS)

cap = cv2.VideoCapture(VIDEO_IN)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_read = cap.get(cv2.CAP_PROP_FPS) or FPS

out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps_read, (w, h))

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    frame_idx += 1

    # YOLO inference
    res = model.predict(source=frame, verbose=False)[0]

    # Parse best ball & hoop
    best_ball = pick(res.boxes, BALL_NAMES)
    best_hoop = pick(res.boxes, HOOP_NAMES)

    ball_t = None
    if best_ball is not None:
        x1, y1, x2, y2 = best_ball.xyxy[0].tolist()
        bx = (x1 + x2) * 0.5
        by = (y1 + y2) * 0.5
        ball_t = (bx, by, float(best_ball.conf))

    hoop_t = None
    if best_hoop is not None:
        x1, y1, x2, y2 = best_hoop.xyxy[0].tolist()
        hoop_t = (x1, y1, x2-x1, y2-y1, float(best_hoop.conf))

    # Update FSM
    out_dict = sd.update(ball=ball_t, hoop=hoop_t)

    # Draw quick YOLO boxes for context (optional)
    if best_ball is not None:
        x1, y1, x2, y2 = map(int, best_ball.xyxy[0].tolist())
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
    if best_hoop is not None:
        x1, y1, x2, y2 = map(int, best_hoop.xyxy[0].tolist())
        cv2.rectangle(frame, (x1,y1), (x2,y2), (100,255,100), 2)

    # FSM overlays (rim cylinder, trace, state label)
    frame = sd.draw_overlays(frame)

    # Event toast
    if "SHOT_MADE" in out_dict["events"]:
        cv2.putText(frame, "SHOT MADE!", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
        print(f"✅ SHOT_MADE @ frame {out_dict['frame']}")
    if "SHOT_MISSED" in out_dict["events"]:
        cv2.putText(frame, "SHOT MISSED", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
        print(f"❌ SHOT_MISSED @ frame {out_dict['frame']}")

    out.write(frame)

cap.release()
out.release()
print("Finished. Makes:", sd.makes, "Misses:", sd.misses, "→", VIDEO_OUT)
