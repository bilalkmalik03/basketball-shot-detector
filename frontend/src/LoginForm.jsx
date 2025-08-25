import React, { useState } from 'react';
import { useAuth } from './AuthContext';

const LoginForm = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ username: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, register } = useAuth();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      let result;
      if (isLogin) {
        result = await login(formData.username, formData.password);
      } else {
        result = await register(formData.username, formData.email, formData.password);
      }

      if (!result.success) {
        setError(result.error);
      }
    } catch (err) {
      setError('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      maxWidth: '400px', 
      margin: '50px auto', 
      padding: '20px',
      border: '1px solid #ddd',
      borderRadius: '8px',
      fontFamily: 'system-ui'
    }}>
      <h2>{isLogin ? 'Login' : 'Register'}</h2>
      
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Username:</label>
          <input
            type="text"
            name="username"
            value={formData.username}
            onChange={handleChange}
            required
            style={{
              width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc'
            }}
          />
        </div>

        {!isLogin && (
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>Email:</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              style={{
                width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc'
              }}
            />
          </div>
        )}

        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Password:</label>
          <input
            type="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
            style={{
              width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc'
            }}
          />
        </div>

        {error && (
          <div style={{ 
            color: 'red', marginBottom: '15px', padding: '8px',
            backgroundColor: '#fee', borderRadius: '4px'
          }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: '100%', padding: '10px', backgroundColor: '#007bff',
            color: 'white', border: 'none', borderRadius: '4px',
            cursor: 'pointer', marginBottom: '10px'
          }}
        >
          {loading ? 'Please wait...' : (isLogin ? 'Login' : 'Register')}
        </button>
      </form>

      <p style={{ textAlign: 'center', marginTop: '15px' }}>
        {isLogin ? "Don't have an account? " : "Already have an account? "}
        <button
          type="button"
          onClick={() => {
            setIsLogin(!isLogin);
            setError('');
            setFormData({ username: '', email: '', password: '' });
          }}
          style={{
            background: 'none', border: 'none', color: '#007bff',
            cursor: 'pointer', textDecoration: 'underline'
          }}
        >
          {isLogin ? 'Register' : 'Login'}
        </button>
      </p>
    </div>
  );
};

export default LoginForm;