# pipeline.py — Kalman + YOLO Shot Detector (headless‑safe, outputs folder)
# Run examples:
#   python src/pipeline.py --input hooptest.mp4 --output outputs/kalman_output.mp4 --show
#   python src/pipeline.py --input hooptest.mp4 --output outputs/kalman_output.mp4

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter


def _clamp_point(x, y, w, h):
    return max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1))


def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


class KalmanBallTracker:
    """Constant-velocity Kalman filter for 2D ball tracking.
    State x = [px, py, vx, vy], Measurements z = [px, py].
    Note: We *use* the filter only to bridge short dropouts. When a detection
    exists, we trust the detection for drawing to avoid visual jitter.
    """

    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State vector
        self.kf.x = np.zeros(4, dtype=float)

        # Transition (F)
        self._set_F(self.dt)

        # Measurement (H)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )

        # Measurement noise (R): lower -> trust detector more
        self.kf.R = np.eye(2, dtype=float) * 4.0  # ~2px std

        # Process noise (Q): small but non-zero; we don't over-smooth
        self.kf.Q = np.eye(4, dtype=float)
        self.kf.Q[0:2, 0:2] *= 0.02
        self.kf.Q[2:4, 2:4] *= 0.005

        # State covariance (P)
        self.kf.P = np.eye(4, dtype=float) * 200.0

        self.initialized = False

    def _set_F(self, dt: float):
        self.kf.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def reset_dt(self, dt: float):
        self.dt = float(dt)
        self._set_F(self.dt)

    def predict(self):
        if not self.initialized:
            return None
        self.kf.predict()
        return self.kf.x[:2].copy()

    def update(self, measurement_xy):
        z = np.asarray(measurement_xy, dtype=float).reshape(2,)
        if not self.initialized:
            self.kf.x[:2] = z
            self.initialized = True
        else:
            self.kf.update(z)
        return self.kf.x[:2].copy()


class KalmanShotDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.ball_tracker = KalmanBallTracker(dt=1.0)

        self.shot_count = 0
        self.ball_positions = deque(maxlen=90)   # raw positions (for debug)
        self.smooth_positions = deque(maxlen=30) # visually smoothed positions

        # EMA for visual smoothing (higher = more responsive, lower = smoother)
        self._ema_beta = 0.6
        self._ema_center = None  # running EMA of (x, y)

        # Hoop state (smoothed)
        self.hoop_center = None  # (x, y)
        self.hoop_size = None    # (w, h)
        self._hoop_alpha = 0.25  # EMA smoothing for hoop bbox

        # Ball box state (for drawing a rectangle even on predicted frames)
        self.last_ball_det_box = None  # (x1, y1, x2, y2) from last detection
        self.ball_box_size = None      # (w, h) smoothed size
        self._ball_box_alpha = 0.3

        # Shot detection state
        self.last_shot_frame = -60
        self.cooldown_frames = 36  # ~0.6s @60fps

        
    def _smooth_hoop(self, center, size):
        if self.hoop_center is None:
            self.hoop_center = center
            self.hoop_size = size
            return
        ax = self._hoop_alpha
        hx, hy = self.hoop_center
        hw, hh = self.hoop_size
        cx, cy = center
        sw, sh = size
        self.hoop_center = ((1 - ax) * hx + ax * cx, (1 - ax) * hy + ax * cy)
        self.hoop_size = ((1 - ax) * hw + ax * sw, (1 - ax) * hh + ax * sh)

    def _smooth_ball_box_size(self, w, h):
        if self.ball_box_size is None:
            self.ball_box_size = (w, h)
        else:
            ax = self._ball_box_alpha
            bw, bh = self.ball_box_size
            self.ball_box_size = ((1 - ax) * bw + ax * w, (1 - ax) * bh + ax * h)

    def _ema_update(self, point):
        """Update the EMA smoother for the trail line and return the smoothed point."""
        if self._ema_center is None:
            self._ema_center = (float(point[0]), float(point[1]))
        else:
            bx = self._ema_beta
            px, py = self._ema_center
            self._ema_center = ( (1 - bx) * px + bx * float(point[0]),
                                 (1 - bx) * py + bx * float(point[1]) )
        return int(round(self._ema_center[0])), int(round(self._ema_center[1]))

    def process_video(self, video_path: str, output_path: str | None = None, show: bool = False):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Update Kalman dt
        self.ball_tracker.reset_dt(1.0 / max(1.0, fps))

        writer = None
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if output_path and not writer.isOpened():
                raise RuntimeError(f"VideoWriter failed to open for: {out_path}")

        frame_num = 0
        frames_since_ball_det = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # ---- Detection
                results = self.model(frame, conf=0.4, verbose=False)
                r = results[0]
                ball_detected = False
                ball_measurement = None

                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_raw = box.cls[0]
                    cls_id = int(cls_raw.item()) if hasattr(cls_raw, "item") else int(cls_raw)
                    label = self.model.names.get(cls_id, str(cls_id)).lower()

                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    if "ball" in label:
                        ball_detected = True
                        ball_measurement = (cx, cy)
                        frames_since_ball_det = 0
                        # save & smooth ball bbox size; draw detection rectangle (green)
                        self.last_ball_det_box = (int(x1), int(y1), int(x2), int(y2))
                        self._smooth_ball_box_size(x2 - x1, y2 - y1)
                        bx1, by1, bx2, by2 = _clamp_box(x1, y1, x2, y2, width, height)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        cv2.putText(frame, "ball", (bx1, max(0, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    elif any(k in label for k in ("hoop", "rim", "basket")):
                        self._smooth_hoop((cx, cy), (x2 - x1, y2 - y1))
                        hx1, hy1, hx2, hy2 = _clamp_box(x1, y1, x2, y2, width, height)
                        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
                        cv2.putText(frame, "hoop", (hx1, max(0, hy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # ---- Decide draw point: use detection directly when available
                draw_point = None
                if ball_detected:
                    # Update KF for state continuity, but trust the detection for drawing
                    self.ball_tracker.update(ball_measurement)
                    draw_point = (ball_measurement[0], ball_measurement[1])
                else:
                    frames_since_ball_det += 1
                    if self.ball_tracker.initialized and frames_since_ball_det <= 10:
                        pred = self.ball_tracker.predict()
                        if pred is not None:
                            draw_point = (pred[0], pred[1])
                    else:
                        draw_point = None

                # ---- Draw & shot logic
                if draw_point is not None:
                    bx, by = _clamp_point(draw_point[0], draw_point[1], width, height)
                    self.ball_positions.append((bx, by))

                    # Only add to trail if ball is actually detected (not predicted)
                    if ball_detected:
                        sbx, sby = self._ema_update((bx, by))
                        self.smooth_positions.append((sbx, sby))
                    else:
                        # Clear trail when we're relying on predictions
                        self.smooth_positions.clear()
                        self._ema_center = None  # Reset EMA state

                    # Draw rectangle for predicted-only frames using last size
                    if (not ball_detected) and self.ball_box_size is not None:
                        bw, bh = self.ball_box_size
                        px1 = bx - bw / 2.0
                        py1 = by - bh / 2.0
                        px2 = bx + bw / 2.0
                        py2 = by + bh / 2.0
                        px1, py1, px2, py2 = _clamp_box(px1, py1, px2, py2, width, height)
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                        cv2.putText(frame, "ball (pred)", (px1, max(0, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Draw center point
                    color = (0, 255, 0) if ball_detected else (0, 255, 255)
                    display_x, display_y = (sbx, sby) if ball_detected and len(self.smooth_positions) > 0 else (bx, by)
                    cv2.circle(frame, (display_x, display_y), 10, color, 2, lineType=cv2.LINE_AA)

                    # Only check for shots when we have actual detections
                    if ball_detected and self.check_shot(display_x, display_y, frame_num):
                        self.shot_count += 1
                        cv2.putText(
                            frame,
                            "SHOT!",
                            (width // 2 - 70, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 0),
                            4,
                        )

                # Trail (only draw when we have actual detections)
                if len(self.smooth_positions) > 1:
                    pts = np.array(self.smooth_positions, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                # UI
                cv2.putText(
                    frame,
                    f"Shots: {self.shot_count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3,
                )
                status = (
                    "Detected"
                    if ball_detected
                    else (
                        f"Predicted ({frames_since_ball_det})"
                        if (self.ball_tracker.initialized and frames_since_ball_det <= 10)
                        else "N/A"
                    )
                )
                cv2.putText(
                    frame,
                    f"Ball: {status}",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 200),
                    2,
                )

                # Rim line
                if self.hoop_center and self.hoop_size:
                    hx, hy = map(int, self.hoop_center)
                    hw, hh = self.hoop_size
                    rim_y = int(hy - 25)
                    cv2.line(
                        frame,
                        (hx - int(hw // 2), rim_y),
                        (hx + int(hw // 2), rim_y),
                        (255, 0, 0),
                        2,
                        lineType=cv2.LINE_AA,
                    )

                if writer is not None:
                    writer.write(frame)

                if show:
                    try:
                        cv2.imshow("Kalman Tracking", frame)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC
                            break
                    except cv2.error:
                        # Fall back to headless mid-run if GUI isn't available
                        show = False

                frame_num += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        return self.shot_count
        


    def check_shot(self, bx: int, by: int, frame_num: int) -> bool:
            """Detect shot by simple top line crossing with proper gating"""
            if self.hoop_center is None or self.hoop_size is None:
                return False
            
            # Cooldown to avoid double counts
            if frame_num - self.last_shot_frame < self.cooldown_frames:
                return False

            hx, hy = self.hoop_center
            hw, hh = self.hoop_size
            
            # Detection line (same as your visual line)
            rim_y = int(hy - 35)
            
            # Horizontal gate - ball must be reasonably centered
            gate_width = 0.6 * hw  # 60% of hoop width
            
            # Need at least 2 positions for crossing detection
            if len(self.smooth_positions) < 2:
                return False
            
            prev_x, prev_y = self.smooth_positions[-2]
            curr_x, curr_y = bx, by
            
            # Check for downward crossing of the rim line
            if prev_y < rim_y <= curr_y:  # Ball crossed from above to below
                if abs(curr_x - hx) <= gate_width:  # Ball is centered enough
                    if curr_y > prev_y:  # Confirm downward motion
                        self.last_shot_frame = frame_num
                        print(f"[SHOT] frame={frame_num} crossed rim_y={rim_y} at x={bx:.1f} (hoop center={hx:.1f})")
                        return True
            
            return False



def main():
    parser = argparse.ArgumentParser(
        description="Kalman + YOLO basketball shot detector (headless‑safe)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path("models/basketball_hoop/weights/best.pt")),
        help="Path to YOLO model (.pt) with 'ball' and 'hoop'/'rim' classes",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="hooptest.mp4",
        help="Input video path (relative to project root is fine)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/kalman_output.mp4",
        help="Output annotated video path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show OpenCV preview window (requires GUI backend)",
    )
    args = parser.parse_args()

    # Resolve project root (assumes this file is at src/pipeline.py)
    BASE_DIR = Path(__file__).resolve().parents[1]
    model_path = (BASE_DIR / args.model).resolve()
    in_path = (BASE_DIR / args.input).resolve()
    out_path = (BASE_DIR / args.output).resolve()

    print(f"Model:  {model_path}")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")

    os.makedirs(out_path.parent, exist_ok=True)

    detector = KalmanShotDetector(str(model_path))
    shots = detector.process_video(str(in_path), str(out_path), show=args.show)
    print(f"Detected {shots} shots using Kalman filtering")


if __name__ == "__main__":
    main()
