import { logout } from '../api/client';

export default function Navbar({ user, view, setView, onLogout }) {
  return (
    <nav className="navbar">
      <div className="navbar-brand" onClick={() => setView('tester')}>
        <span className="brand-icon">🛡</span>
        <span className="brand-text">PromptShield</span>
      </div>

      <div className="navbar-links">
        <button
          className={`nav-link ${view === 'tester' ? 'active' : ''}`}
          onClick={() => setView('tester')}
        >
          Prompt Tester
        </button>

        {user && (
          <button
            className={`nav-link ${view === 'history' ? 'active' : ''}`}
            onClick={() => setView('history')}
          >
            History
          </button>
        )}
      </div>

      <div className="navbar-user">
        {user ? (
          <>
            <span className="user-email">{user.email}</span>
            <button
              className="btn-logout"
              onClick={() => { logout(); onLogout(); }}
            >
              Sign Out
            </button>
          </>
        ) : (
          <button
            className={`nav-link ${view === 'login' ? 'active' : ''}`}
            onClick={() => setView('login')}
          >
            Sign In
          </button>
        )}
      </div>
    </nav>
  );
}
