import { useState } from 'react';
import { predict } from '../api/client';

export default function PromptTester() {
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleCheck = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await predict(prompt);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isInjection = result?.label === 'PROMPT_INJECTION';
  const pct = result ? (result.confidence * 100).toFixed(1) : 0;

  return (
    <div className="card prompt-tester">
      <h2>Prompt Tester</h2>

      <form onSubmit={handleCheck}>
        <textarea
          id="prompt-input"
          rows="5"
          placeholder="Type or paste a prompt to check..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          maxLength={4000}
        />
        <div className="textarea-footer">
          <span className="char-count">{prompt.length} / 4000</span>
          <button type="submit" disabled={loading || !prompt.trim()} id="check-btn">
            {loading ? 'Checking...' : 'Check Prompt'}
          </button>
        </div>
      </form>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className={`result-card ${isInjection ? 'result-danger' : 'result-safe'}`}>
          <div className="result-label">
            <span className={`badge ${isInjection ? 'badge-danger' : 'badge-safe'}`}>
              {result.label === 'PROMPT_INJECTION' ? '⚠ PROMPT INJECTION' : '✓ SAFE'}
            </span>
          </div>

          <div className="confidence-section">
            <span className="confidence-text">Confidence: {pct}%</span>
            <div className="confidence-bar-track">
              <div
                className={`confidence-bar-fill ${isInjection ? 'fill-danger' : 'fill-safe'}`}
                style={{ width: pct + '%' }}
              />
            </div>
          </div>
        </div>
      )}

      <div className="caveat">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 2.5a1 1 0 110 2 1 1 0 010-2zM6.5 7h3l-.5 5h-2L6.5 7z"/>
        </svg>
        <span>
          Statistical classifier — not a guarantee.
          Review flagged prompts before blocking automatically.
        </span>
      </div>
    </div>
  );
}
