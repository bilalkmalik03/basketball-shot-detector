import React from "react";
import { useAuth } from "./AuthContext";
import LoginForm from "./LoginForm";
import ShotUploader from "./uploader";

export default function App() {
  const { user, loading, logout } = useAuth();

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        fontFamily: 'system-ui'
      }}>
        Loading...
      </div>
    );
  }

  if (!user) {
    return <LoginForm />;
  }

  return (
    <div>
      {/* Simple header with user info and logout */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        padding: '10px 20px',
        backgroundColor: '#f8f9fa',
        borderBottom: '1px solid #ddd'
      }}>
        <span>Welcome, {user.username}!</span>
        <button
          onClick={logout}
          style={{
            padding: '5px 15px',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Logout
        </button>
      </div>

      {/* Your existing shot uploader */}
      <ShotUploader />
    </div>
  );
}