# basketball_shot_detector.py
from ultralytics import YOLO
import cv2
import os
from collections import deque
from pathlib import Path

class ShotDetector:
    def __init__(self,
                 model_path='models/basketball_hoop/weights/best.pt',
                 infer_conf=0.25,          # more sensitive (good for small ball)
                 infer_imgsz=768,          # higher res helps small objects
                 cooldown_frames=15):      # avoid double-counting
        model_path = str(model_path)
        assert Path(model_path).exists(), f"Model not found: {model_path}"
        print(">> Loading weights:", model_path)

        self.model = YOLO(model_path)
        self.infer_conf = infer_conf
        self.infer_imgsz = infer_imgsz
        self.cooldown_frames = cooldown_frames
        self.cooldown = 0

        # Map class IDs robustly (handles 'ball' vs 'Basketball' vs 'sports ball', etc.)
        names = {i: n.lower() for i, n in self.model.names.items()}
        self.BALL_IDS = {i for i, n in names.items() if n in ('ball', 'basketball', 'sports ball')}
        self.HOOP_IDS = {i for i, n in names.items() if n in ('hoop', 'rim', 'basket', 'backboard')}
        print(">> Class map:", self.model.names)
        print(">> BALL_IDS:", self.BALL_IDS, "HOOP_IDS:", self.HOOP_IDS)

        self.shot_count = 0
        self.ball_positions = deque(maxlen=30)
        self.hoop_position = None

    def process_video(self, video_path, output_path=None, show=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                # Fallback for some Windows setups
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path = os.path.splitext(output_path)[0] + '.avi'
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO with tuned inference params
            results = self.model(frame, conf=self.infer_conf, imgsz=self.infer_imgsz)

            # Clear hoop for this frame (we'll update if we see one)
            # (optional: you can keep last hoop if not detected this frame)
            # self.hoop_position = None

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if cls_id in self.BALL_IDS:
                        self.ball_positions.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)

                    elif cls_id in self.HOOP_IDS:
                        self.hoop_position = (cx, cy)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Check for shot (with cooldown)
            if self.cooldown > 0:
                self.cooldown -= 1
            elif self.check_shot():
                self.shot_count += 1
                self.cooldown = self.cooldown_frames
                cv2.putText(frame, "SHOT MADE!", (width//2 - 100, height//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # HUD
            cv2.putText(frame, f"Shots: {self.shot_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            if show:
                cv2.imshow('Basketball Detection', frame)
                if cv2.waitKey(1) == 27:  # ESC to quit
                    break

            if out is not None:
                out.write(frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, Shots: {self.shot_count}")

        cap.release()
        if out is not None:
            out.release()
        if show:
            cv2.destroyAllWindows()

        return self.shot_count

    def check_shot(self):
        """Simple shot detection logic: near hoop AND moving downward."""
        if not self.hoop_position or len(self.ball_positions) < 10:
            return False

        hx, hy = self.hoop_position
        # look at last few points of the trail
        start = max(1, len(self.ball_positions) - 5)
        for i in range(start, len(self.ball_positions)):
            bx, by = self.ball_positions[i]
            pbx, pby = self.ball_positions[i - 1]

            near_hoop = (abs(bx - hx) < 30) and (abs(by - hy) < 30)
            moving_down = by > pby
            if near_hoop and moving_down:
                self.ball_positions.clear()  # avoid immediate re-trigger from the same pass
                return True
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/basketball_hoop/weights/best.pt")
    parser.add_argument("--source", default="test_video.mp4")
    parser.add_argument("--out", default="output_detected.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--no-show", action="store_true", help="run headless (no imshow)")
    args = parser.parse_args()

    det = ShotDetector(model_path=args.model, infer_conf=args.conf, infer_imgsz=args.imgsz)
    shots = det.process_video(args.source, args.out, show=not args.no_show)
    print(f"Total shots detected: {shots}\nSaved: {args.out}")
