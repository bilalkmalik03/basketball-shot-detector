import React from "react";
import ReactDOM from "react-dom/client";
import ShotUploader from "./uploader.jsx";
import { AuthProvider } from "./AuthContext.jsx";
import App from "./App.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>
);