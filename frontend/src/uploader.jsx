import { useState } from "react";

export default function ShotUploader() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null); // { shots, video_url }

  const handleFile = (e) => {
    setFile(e.target.files?.[0] || null);
    setResult(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) return;
    setProcessing(true);
    setError("");
    setResult(null);

    try {
      const form = new FormData();
      form.append("video", file);

      // use relative URL -> Vite proxy forwards to Flask
      const res = await fetch("/api/process", { 
        method: "POST", 
        body: form,
        credentials: 'include' // ADDED: Important for session cookies
      });

      if (!res.ok) {
        // ADDED: Handle authentication errors
        if (res.status === 401) {
          // Session expired - refresh the page to show login
          alert('Session expired. Please log in again.');
          window.location.reload();
          return;
        }
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `Server error: ${res.status}`);
      }
      const data = await res.json();
      // add cache-busting param so <video> reloads fresh file
      data.video_url = `${data.video_url}?t=${Date.now()}`;
      setResult(data);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h2>Basketball Shot Detector</h2>

      <input type="file" accept="video/*" onChange={handleFile} disabled={processing} />

      <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
        <button onClick={handleUpload} disabled={!file || processing}>
          {processing ? "Processing..." : "Upload & Detect"}
        </button>
        {file && <span>{file.name}</span>}
      </div>

      {error && <p style={{ color: "crimson", marginTop: 12 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 24 }}>
          <p><strong>Shots detected:</strong> {result.shots}</p>
          <video
            key={result.video_url}           // force reload on new URL
            src={result.video_url}           // relative URL (proxied)
            controls
            style={{ width: "100%", maxHeight: 480, background: "#000" }}
          />
          <p style={{ fontSize: 12, color: "#666" }}>Saved from backend: {result.video_url}</p>
        </div>
      )}
    </div>
  );
}