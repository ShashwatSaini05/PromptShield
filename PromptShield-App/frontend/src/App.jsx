import { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import PromptTester from './components/PromptTester';
import LoginForm from './components/LoginForm';
import HistoryTable from './components/HistoryTable';
import { getMe, isLoggedIn } from './api/client';

export default function App() {
  const [user, setUser] = useState(null);
  const [view, setView] = useState('tester');

  // Try to restore session on mount
  useEffect(() => {
    if (isLoggedIn()) {
      getMe()
        .then(setUser)
        .catch(() => setUser(null));
    }
  }, []);

  const handleAuth = () => {
    getMe().then((u) => {
      setUser(u);
      setView('tester');
    });
  };

  const handleLogout = () => {
    setUser(null);
    setView('tester');
  };

  return (
    <>
      <Navbar user={user} view={view} setView={setView} onLogout={handleLogout} />
      <main className="container">
        {view === 'tester' && <PromptTester />}
        {view === 'login' && <LoginForm onAuth={handleAuth} />}
        {view === 'history' && user && <HistoryTable />}
        {view === 'history' && !user && (
          <div className="card">
            <p>Please sign in to view your prediction history.</p>
            <button onClick={() => setView('login')}>Sign In</button>
          </div>
        )}
      </main>
    </>
  );
}
