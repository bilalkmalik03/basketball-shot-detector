# 🏀 Basketball Shot Tracker

An end-to-end **basketball shot detection and tracking system** powered by **computer vision (YOLOv8)**, **Kalman filtering**, and a **Flask backend**, with video storage on **AWS S3**.
This project detects basketballs and hoops in real-time, tracks trajectories, and determines whether a shot is made.

---

## 🚀 Features

* 🎯 **Ball & Hoop Detection**: Uses YOLOv8 for robust object detection.
* 📈 **Trajectory Tracking**: Kalman filter smooths ball movement and predicts shot paths.
* 🏆 **Shot Detection**: Determines whether a ball passes through the hoop.
* 🌐 **Web Backend**: Flask API for processing video uploads.
* ☁️ **Cloud Storage**: Video uploads are stored in **AWS S3**.
* 🖥️ **Frontend Integration (Planned)**: For uploading videos and viewing results in browser.

---

## 🛠️ Tech Stack

* **Python 3.11**
* **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
* **ML / Tracking**: FilterPy (Kalman filter), NumPy
* **Backend**: Flask, Flask-CORS
* **Cloud**: AWS S3 (Boto3 SDK)
* **Other**: Pandas, Matplotlib (analytics/visualization)

---

## 📂 Project Structure

```
basketball-shot-tracker/
│── backend/
│   ├── app.py           # Flask backend API
│   ├── detect.py        # YOLO detection + Kalman filter pipeline
│   ├── pipeline.py      # Tracking + shot validation
│   └── requirements.txt
│
│── dataset/             # Training/validation data (YOLO format)
│── models/              # YOLO weights (hoop + ball detection)
│── uploads/             # Temporary video uploads
│── README.md
```

---

## ⚡ Installation

1. **Clone the repo**

```bash
git clone https://github.com/USERNAME/basketball-shot-tracker.git
cd basketball-shot-tracker/backend
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up AWS credentials**
   Make sure you configure your AWS CLI or add credentials in environment variables:

```bash
aws configure
```

Environment variables needed:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_S3_BUCKET=your_bucket_name
```

---

## ▶️ Usage

1. **Run the backend**

```bash
python app.py
```

2. **Upload a video** (via frontend or API)
   Example using `curl`:

```bash
curl -X POST -F "file=@uploads/test.mp4" http://127.0.0.1:5000/process
```

3. **Results**

* Annotated video stored in `uploads/processed/`
* Shot detection results returned as JSON
* Original video uploaded to **AWS S3**

---

## 📊 Example Output

* ✅ Ball detected and tracked with green/yellow trajectory line
* ✅ Hoop detected with bounding box
* ✅ JSON response:

```json
{
  "shots_detected": 3,
  "made_shots": 2,
  "missed_shots": 1,
  "s3_url": "https://s3.amazonaws.com/your-bucket/video.mp4"
}
```

---

## 🌍 Deployment

* **Local**: Flask runs on `http://127.0.0.1:5000/`
* **Cloud (Future)**:

  * Deploy Flask backend on AWS EC2, Elastic Beanstalk, or Docker
  * Store and stream processed videos via AWS S3 + CloudFront
  * Deploy frontend on Vercel/Netlify

---

## 📌 Roadmap

* [x] Ball & hoop detection with YOLO
* [x] Kalman filter trajectory tracking
* [x] Flask backend + AWS S3 integration
* [ ] Frontend web app (React + Tailwind)
* [ ] Real-time live camera input
* [ ] Analytics dashboard for shot percentage over time

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

---

