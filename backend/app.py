from __future__ import annotations
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, Response, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import subprocess, shutil
import os, uuid, sys
import sqlite3
from functools import wraps
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

# Optional: OpenCV is only used to quickly test readability of uploads
try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

# Load environment variables
load_dotenv()

# ---- Paths / Imports ----
BASE_DIR = Path(__file__).resolve().parent.parent  # project root (../ from this file)
sys.path.append(str(BASE_DIR))  # allow "src" imports from project root

from src.pipeline import KalmanShotDetector  # ensure this exists

UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "models" / "basketball_hoop" / "weights" / "best.pt"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Initialize S3 client
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    # Test S3 connection on startup
    if S3_BUCKET_NAME:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"Connected to S3 bucket: {S3_BUCKET_NAME}")
        S3_AVAILABLE = True
    else:
        print("S3_BUCKET_NAME not configured")
        S3_AVAILABLE = False
except Exception as e:
    print(f"S3 not available: {e}")
    S3_AVAILABLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

# SECRET KEY FOR SESSIONS - CHANGE THIS TO SOMETHING RANDOM AND SECURE!
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this-to-something-random')

# Accept both with/without trailing slash
app.url_map.strict_slashes = False

# CORS for local dev (adjust as needed)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
            ]
        }
    },
    supports_credentials=True  # Important for session cookies
)

# ---- Authentication Setup ----
def init_db():
    """Initialize the database with users and shots tables"""
    conn = sqlite3.connect('basketball_tracker.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Shot sessions table (to track user's shot tracking sessions)
    c.execute('''CREATE TABLE IF NOT EXISTS shot_sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  shots_detected INTEGER NOT NULL,
                  video_filename TEXT,
                  s3_url TEXT,
                  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def update_db_schema():
    """Add s3_url column to existing shot_sessions table"""
    try:
        conn = sqlite3.connect('basketball_tracker.db')
        c = conn.cursor()
        
        # Check if column exists
        c.execute("PRAGMA table_info(shot_sessions)")
        columns = [column[1] for column in c.fetchall()]
        
        if 's3_url' not in columns:
            c.execute('ALTER TABLE shot_sessions ADD COLUMN s3_url TEXT')
            conn.commit()
            print("Added s3_url column to shot_sessions table")
            
        conn.close()
    except Exception as e:
        print(f"Failed to update database schema: {e}")

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Initialize database when app starts
init_db()
update_db_schema()

# Load your detector once
try:
    detector = KalmanShotDetector(str(MODEL_PATH))
except Exception as e:
    print(f"[MODEL] Failed to initialize KalmanShotDetector: {e}")
    detector = None

# ---- S3 Helper Functions ----
def upload_file_to_s3(file_path, s3_key):
    """Upload a file to S3 and return the URL"""
    try:
        s3_client.upload_file(
            str(file_path), 
            S3_BUCKET_NAME, 
            s3_key
        )
        return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"Failed to upload {file_path} to S3: {e}")
        return None

def delete_local_file(file_path):
    """Delete local file after successful S3 upload"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Failed to delete local file {file_path}: {e}")

# ---- Helpers (unchanged) ----
def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _can_open_with_cv2(path: Path) -> bool:
    """Quick sanity check if OpenCV can open at least one frame."""
    if cv2 is None:
        return True  # don't block if cv2 isn't available
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def _transcode_to_mp4(input_path: Path, output_path: Path) -> bool:
    """Full transcode to H.264 (yuv420p) + AAC MP4 with +faststart for web playback."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        str(output_path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _ensure_web_mp4(input_path: Path, output_path: Path) -> bool:
    """
    Force a rock‑solid, browser‑friendly MP4:
      - H.264 (Baseline, level 3.0), yuv420p
      - Constant frame rate 30 FPS, GOP=60, no scene‑cut bias
      - AAC audio if present
      - +faststart so moov atom is at the beginning (progressive streaming)
    """
    if not _ffmpeg_available():
        return True  # can't fix without ffmpeg; serve as-is

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),

        # Video
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level:v", "3.0",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-g", "60",
        "-sc_threshold", "0",

        # Audio (copy to AAC if present; ok if no audio)
        "-c:a", "aac",
        "-ar", "48000",
        "-b:a", "128k",

        # Fast start
        "-movflags", "+faststart",

        # Map first video, optional audio
        "-map", "0:v:0",
        "-map", "0:a?",

        str(output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


# ---- Error handlers ----
@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large."}), 413

# ---- Authentication Routes ----
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    
    # Basic validation
    if not username or len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    
    if not email or '@' not in email:
        return jsonify({'error': 'Valid email required'}), 400
        
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    # Hash password
    password_hash = generate_password_hash(password)
    
    try:
        conn = sqlite3.connect('basketball_tracker.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)', 
                 (username, email, password_hash))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Auto-login after registration
        session['user_id'] = user_id
        session['username'] = username
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {'id': user_id, 'username': username, 'email': email}
        }), 201
        
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return jsonify({'error': 'Username already exists'}), 409
        elif 'email' in str(e):
            return jsonify({'error': 'Email already exists'}), 409
        else:
            return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    conn = sqlite3.connect('basketball_tracker.db')
    c = conn.cursor()
    c.execute('SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?', 
              (username, username))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[3], password):
        session['user_id'] = user[0]
        session['username'] = user[1]
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {'id': user[0], 'username': user[1], 'email': user[2]}
        })
    else:
        return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/me', methods=['GET'])
@login_required
def get_current_user():
    conn = sqlite3.connect('basketball_tracker.db')
    c = conn.cursor()
    c.execute('SELECT id, username, email FROM users WHERE id = ?', (session['user_id'],))
    user = c.fetchone()
    
    if not user:
        session.clear()
        return jsonify({'error': 'User not found'}), 404
    
    # Get user's shot history
    c.execute('''SELECT COUNT(*) as total_sessions, SUM(shots_detected) as total_shots 
                 FROM shot_sessions WHERE user_id = ?''', (session['user_id'],))
    stats = c.fetchone()
    conn.close()
    
    return jsonify({
        'user': {
            'id': user[0], 
            'username': user[1], 
            'email': user[2]
        },
        'stats': {
            'total_sessions': stats[0] or 0,
            'total_shots': stats[1] or 0
        }
    })

# ---- Main Routes (updated with S3 integration) ----
@app.post("/api/process")
@login_required
def process_video():
    if detector is None:
        return jsonify({"error": "Detector not initialized."}), 500

    if "video" not in request.files:
        return jsonify({"error": "No file field 'video'"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    orig_name = secure_filename(file.filename)
    uid = uuid.uuid4().hex[:8]
    in_path = UPLOADS_DIR / f"{uid}_{orig_name or 'upload.mp4'}"
    out_path = OUTPUTS_DIR / f"{uid}_output.mp4"

    # Save upload
    file.save(str(in_path))
    if not in_path.exists() or in_path.stat().st_size == 0:
        return jsonify({"error": "Upload failed"}), 500

    # If OpenCV can't read the upload, transcode it to a temp mp4 first
    src_for_detection = in_path
    if not _can_open_with_cv2(in_path):
        if not _ffmpeg_available():
            return jsonify({"error": "Unsupported codec and ffmpeg missing"}), 415
        trans_mp4 = UPLOADS_DIR / f"{uid}_input_transcoded.mp4"
        ok = _transcode_to_mp4(in_path, trans_mp4)
        if not ok:
            return jsonify({"error": "Transcode failed"}), 415
        src_for_detection = trans_mp4

    # Run your detector to produce an output video
    try:
        shots = detector.process_video(str(src_for_detection), str(out_path), show=False)
    except Exception as e:
        # Clean up before returning
        try:
            if src_for_detection != in_path:
                src_for_detection.unlink(missing_ok=True)
            in_path.unlink(missing_ok=True)
        finally:
            return jsonify({"error": str(e)}), 500

    # Ensure the generated output is web-playable
    try:
        fixed_path = out_path.with_name(out_path.stem + "_web.mp4")
        ok = _ensure_web_mp4(out_path, fixed_path)
        if ok and fixed_path.exists() and fixed_path.stat().st_size > 0:
            out_path.unlink(missing_ok=True)
            out_path = fixed_path
    except Exception:
        # best-effort; serve original out_path if fix fails
        pass

    # Upload processed video to S3
    video_url = f"/outputs/{out_path.name}"  # Fallback to local serving
    s3_url = None
    
    if S3_AVAILABLE and out_path.exists():
        s3_key = f"processed/{uid}/{out_path.name}"
        s3_url = upload_file_to_s3(out_path, s3_key)
        
        if s3_url:
            video_url = s3_url
            # Optionally delete local file after successful S3 upload
            # delete_local_file(out_path)

    # Save shot session to database (now includes S3 URL)
    try:
        conn = sqlite3.connect('basketball_tracker.db')
        c = conn.cursor()
        c.execute('''INSERT INTO shot_sessions 
                     (user_id, shots_detected, video_filename, s3_url) 
                     VALUES (?, ?, ?, ?)''',
                 (session['user_id'], int(shots), out_path.name, s3_url))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to save shot session: {e}")

    # Cleanup inputs to save space
    try:
        if src_for_detection != in_path:
            src_for_detection.unlink(missing_ok=True)
        in_path.unlink(missing_ok=True)
    except Exception:
        pass

    return jsonify({
        "shots": int(shots),
        "video_url": video_url,  # Now either S3 URL or local fallback
        "s3_url": s3_url,
        "id": uid,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), 201


@app.get("/outputs/<path:filename>")
def get_output(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "Not found"}), 404

    file_size = file_path.stat().st_size
    range_header = request.headers.get("Range")

    def partial_response(start: int, end: int) -> Response:
        length = end - start + 1
        with open(file_path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        resp = Response(data, 206, mimetype="video/mp4")
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = str(length)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    if not range_header:
        with open(file_path, "rb") as f:
            data = f.read()
        resp = Response(data, 200, mimetype="video/mp4")
        resp.headers["Content-Length"] = str(file_size)
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    try:
        _, rng = range_header.split("=")
        start_str, end_str = (rng.split("-") + [""])[:2]
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        start = max(0, start)
        end = min(end, file_size - 1)
        if start > end:
            return Response(status=416)
    except Exception:
        return Response(status=416)

    return partial_response(start, end)


if __name__ == "__main__":
    # Run: python web/app.py
    app.run(host="0.0.0.0", port=5000, debug=True)