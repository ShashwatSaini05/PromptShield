import { useState } from 'react';
import { login, signup } from '../api/client';

export default function LoginForm({ onAuth }) {
  const [isSignup, setIsSignup] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      if (isSignup) {
        await signup(email, password);
        setSuccess('Account created! Logging you in...');
        // auto-login after signup
        await login(email, password);
      } else {
        await login(email, password);
      }
      onAuth();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card login-form">
      <h2>{isSignup ? 'Create Account' : 'Sign In'}</h2>

      <form onSubmit={handleSubmit}>
        <label htmlFor="email-input">Email</label>
        <input
          id="email-input"
          type="email"
          placeholder="you@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <label htmlFor="password-input">Password</label>
        <input
          id="password-input"
          type="password"
          placeholder={isSignup ? 'Min 8 characters' : 'Your password'}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={isSignup ? 8 : undefined}
        />

        {error && <div className="alert alert-error">{error}</div>}
        {success && <div className="alert alert-success">{success}</div>}

        <button type="submit" disabled={loading} id="auth-submit-btn">
          {loading ? 'Please wait...' : isSignup ? 'Sign Up' : 'Sign In'}
        </button>
      </form>

      <p className="toggle-auth">
        {isSignup ? 'Already have an account?' : "Don't have an account?"}
        <button
          type="button"
          className="link-btn"
          onClick={() => { setIsSignup(!isSignup); setError(''); setSuccess(''); }}
        >
          {isSignup ? 'Sign In' : 'Sign Up'}
        </button>
      </p>
    </div>
  );
}
