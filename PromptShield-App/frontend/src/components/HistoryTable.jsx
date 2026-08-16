import { useState, useEffect } from 'react';
import { getHistory } from '../api/client';

export default function HistoryTable() {
  const [data, setData] = useState(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const pageSize = 20;

  const fetchHistory = async (p) => {
    setLoading(true);
    setError('');
    try {
      const result = await getHistory(p, pageSize);
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory(page);
  }, [page]);

  const truncate = (text, len = 80) =>
    text.length > len ? text.slice(0, len) + '...' : text;

  const formatDate = (iso) => {
    const d = new Date(iso);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
  };

  return (
    <div className="card history-table">
      <h2>Prediction History</h2>

      {error && <div className="alert alert-error">{error}</div>}

      {loading && !data && <p className="loading-text">Loading...</p>}

      {data && data.items.length === 0 && (
        <p className="empty-text">No predictions yet. Try the Prompt Tester!</p>
      )}

      {data && data.items.length > 0 && (
        <>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Prompt</th>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item) => (
                  <tr key={item.id}>
                    <td className="prompt-cell" title={item.prompt_text}>
                      {truncate(item.prompt_text)}
                    </td>
                    <td>
                      <span className={
                        item.predicted_label === 'PROMPT_INJECTION'
                          ? 'badge badge-danger'
                          : 'badge badge-safe'
                      }>
                        {item.predicted_label === 'PROMPT_INJECTION' ? '⚠ INJECTION' : '✓ SAFE'}
                      </span>
                    </td>
                    <td>{(item.confidence * 100).toFixed(1)}%</td>
                    <td className="time-cell">{formatDate(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <button
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
            >
              ← Prev
            </button>
            <span>
              Page {data.page} of {data.pages} ({data.total} total)
            </span>
            <button
              disabled={page >= data.pages}
              onClick={() => setPage(page + 1)}
            >
              Next →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
