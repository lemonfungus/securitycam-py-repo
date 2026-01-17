import { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import './App.css';

// ==================== MOCK DATA ====================
const MOCK_CAMERAS = [
  { id: 1, name: 'Front Entrance', status: 'online' },
  { id: 2, name: 'Parking Lot', status: 'online' },
  { id: 3, name: 'Back Door', status: 'offline' },
  { id: 4, name: 'Lobby', status: 'online' },
];

const MOCK_ALERTS = [
  { id: 1, type: 'critical', message: 'Person detected in restricted area', time: '21:38', camera: 'Front Entrance' },
  { id: 2, type: 'warning', message: 'Motion detected after hours', time: '21:35', camera: 'Parking Lot' },
  { id: 3, type: 'info', message: 'Camera reconnected', time: '21:30', camera: 'Back Door' },
  { id: 4, type: 'warning', message: 'Low light conditions', time: '21:25', camera: 'Lobby' },
];

// ==================== LOGIN COMPONENT ====================
function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const res = await axios.post('http://localhost:3000/api/login', { username, password });
      if (res.data.success) {
        onLogin(username);
      }
    } catch (err) {
      setError('Invalid credentials. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-logo">
          <div className="login-logo-icon">🔒</div>
          <h1>INCOGNITOVISION</h1>
          <p>Secure Surveillance System</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          {error && <div className="login-error">{error}</div>}

          <div className="input-group">
            <label>Username</label>
            <input
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div className="input-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? (
              <>
                <div className="spinner"></div>
                Authenticating...
              </>
            ) : (
              'Access System'
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

// ==================== HEADER COMPONENT ====================
function Header({ user, onLogout }) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formattedTime = time.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }) + ' ' + time.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-brand-icon">🔒</div>
        <h1>INCOGNITOVISION</h1>
      </div>
      <div className="header-info">
        <span className="header-time">{formattedTime}</span>
        <div className="header-user">
          <div className="header-user-avatar">👤</div>
          <span>{user}</span>
        </div>
      </div>
    </header>
  );
}

// ==================== SIDEBAR COMPONENT ====================
function Sidebar({ view, setView, cameras, selectedCamera, setSelectedCamera, onLogout }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-section">
        <div className="sidebar-section-title">Cameras</div>
        <div className="camera-list">
          {cameras.map((camera) => (
            <div
              key={camera.id}
              className={`camera-item ${selectedCamera?.id === camera.id ? 'selected' : ''}`}
              onClick={() => setSelectedCamera(camera)}
            >
              <div className={`camera-status ${camera.status}`}></div>
              <span className="camera-name">{camera.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section-title">Navigation</div>
        <nav className="sidebar-nav">
          <button
            className={`nav-item ${view === 'live' ? 'active' : ''}`}
            onClick={() => setView('live')}
          >
            <span className="nav-item-icon">📺</span>
            <span>Live View</span>
          </button>
          <button
            className={`nav-item ${view === 'logs' ? 'active' : ''}`}
            onClick={() => setView('logs')}
          >
            <span className="nav-item-icon">📂</span>
            <span>Incident Logs</span>
          </button>
          <button
            className={`nav-item ${view === 'settings' ? 'active' : ''}`}
            onClick={() => setView('settings')}
          >
            <span className="nav-item-icon">⚙️</span>
            <span>Settings</span>
          </button>
        </nav>
      </div>

      <div className="sidebar-section" style={{ marginTop: 'auto' }}>
        <nav className="sidebar-nav">
          <button className="nav-item nav-item-danger" onClick={onLogout}>
            <span className="nav-item-icon">🚪</span>
            <span>Logout</span>
          </button>
        </nav>
      </div>
    </aside>
  );
}

// ==================== LIVE VIEW COMPONENT ====================
function LiveView({ camera }) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') setIsFullscreen(false);
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, []);

  return (
    <div className="live-view">
      <div className="live-header">
        <div className="live-title">
          <h2>{camera?.name || 'Live Monitor'}</h2>
          <div className="live-badge">
            <div className="live-badge-dot"></div>
            Live
          </div>
        </div>
        <div className="live-controls">
          <button className="btn-icon" onClick={toggleFullscreen} title="Toggle Fullscreen">
            {isFullscreen ? '⛶' : '⛶'}
          </button>
          <button className="btn-icon" title="Take Snapshot">📷</button>
          <button className="btn-icon" title="Record">⏺</button>
        </div>
      </div>

      <div className={`video-container ${isFullscreen ? 'fullscreen' : ''}`}>
        <img
          src="http://localhost:5000/video_feed"
          alt="Live Feed"
          className="video-feed"
        />
        <div className="video-overlay">
          <div className="video-overlay-item">
            📍 {camera?.name || 'Camera 1'}
          </div>
          <div className="video-overlay-item recording">
            REC
          </div>
        </div>
        {isFullscreen && (
          <button
            className="btn-icon"
            style={{ position: 'absolute', top: 20, right: 20 }}
            onClick={() => setIsFullscreen(false)}
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}

// ==================== LOGS VIEW COMPONENT ====================
function LogsView({ logs, fetchLogs }) {
  const [search, setSearch] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  useEffect(() => {
    fetchLogs();
  }, []);

  const filteredLogs = useMemo(() => {
    return logs.filter((log) => {
      // Search filter
      const matchesSearch = !search ||
        log.class?.toLowerCase().includes(search.toLowerCase()) ||
        log.timestamp?.toLowerCase().includes(search.toLowerCase());

      // Date filter
      let matchesDate = true;
      if (startDate || endDate) {
        const logDate = log.timestamp ? log.timestamp.split(' ')[0] : ''; // Extract YYYY-MM-DD
        if (startDate && logDate < startDate) {
          matchesDate = false;
        }
        if (endDate && logDate > endDate) {
          matchesDate = false;
        }
      }

      return matchesSearch && matchesDate;
    });
  }, [logs, search, startDate, endDate]);

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [search, startDate, endDate]);

  const paginatedLogs = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredLogs.slice(start, start + itemsPerPage);
  }, [filteredLogs, currentPage]);

  const totalPages = Math.ceil(filteredLogs.length / itemsPerPage) || 1;

  const getSeverity = (className) => {
    const critical = ['person', 'weapon', 'fire'];
    const warning = ['motion', 'vehicle'];
    if (critical.some(c => className?.toLowerCase().includes(c))) return 'critical';
    if (warning.some(w => className?.toLowerCase().includes(w))) return 'warning';
    return 'info';
  };

  const exportToCSV = () => {
    const headers = ['Timestamp', 'Class', 'Severity'];
    const rows = filteredLogs.map(log => [
      log.timestamp,
      log.class,
      getSeverity(log.class)
    ]);
    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `incident_logs_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="logs-view">
      <div className="logs-header">
        <div className="logs-title">
          <h2>Incident Logs</h2>
          <p>{filteredLogs.length} incidents recorded</p>
        </div>
        <div className="logs-filters">
          <div className="search-box">
            <span className="search-box-icon">🔍</span>
            <input
              type="text"
              placeholder="Search incidents..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div className="date-filter">
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
            <span>to</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <button className="btn btn-secondary" onClick={exportToCSV}>
            📥 Export CSV
          </button>
        </div>
      </div>

      <div className="logs-table-container">
        {paginatedLogs.length > 0 ? (
          <>
            <table className="logs-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Detection</th>
                  <th>Severity</th>
                  <th>Camera</th>
                </tr>
              </thead>
              <tbody>
                {paginatedLogs.map((log, i) => (
                  <tr key={i}>
                    <td>{log.timestamp}</td>
                    <td>{log.class}</td>
                    <td>
                      <span className={`severity-badge ${getSeverity(log.class)}`}>
                        {getSeverity(log.class)}
                      </span>
                    </td>
                    <td>Camera 1</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="logs-pagination">
              <div className="logs-pagination-info">
                Showing {(currentPage - 1) * itemsPerPage + 1} to {Math.min(currentPage * itemsPerPage, filteredLogs.length)} of {filteredLogs.length}
              </div>
              <div className="logs-pagination-controls">
                <button
                  className="btn-pagination"
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  ‹
                </button>
                <button
                  className="btn-pagination"
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                >
                  ›
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">📋</div>
            <div className="empty-state-title">No incidents found</div>
            <div className="empty-state-text">
              {search ? 'Try adjusting your search filters' : 'No incidents have been recorded yet'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ==================== ALERTS PANEL COMPONENT ====================
function AlertsPanel({ alerts, dismissAlert }) {

  return (
    <aside className="alerts-panel">
      <div className="alerts-header">
        <div className="alerts-title">
          <span>🔔 Alerts</span>
          {alerts.length > 0 && (
            <span className="alerts-count">{alerts.length}</span>
          )}
        </div>
      </div>
      <div className="alerts-list">
        {alerts.length > 0 ? (
          alerts.map((alert) => (
            <div key={alert.id} className={`alert-item ${alert.type}`}>
              <div className="alert-item-header">
                <span className="alert-item-type">
                  {alert.type === 'critical' ? '🔴' : alert.type === 'warning' ? '⚠️' : 'ℹ️'} {alert.type}
                </span>
                <span className="alert-item-time">{alert.time}</span>
              </div>
              <div className="alert-item-message">{alert.message}</div>
              <button
                className="alert-item-dismiss"
                onClick={() => dismissAlert(alert.id)}
              >
                Dismiss
              </button>
            </div>
          ))
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">✅</div>
            <div className="empty-state-title">All clear</div>
            <div className="empty-state-text">No active alerts</div>
          </div>
        )}
      </div>
    </aside>
  );
}

// ==================== SETTINGS VIEW COMPONENT ====================
function SettingsView() {
  return (
    <div className="logs-view">
      <div className="logs-header">
        <div className="logs-title">
          <h2>Settings</h2>
          <p>Configure your surveillance system</p>
        </div>
      </div>
      <div className="logs-table-container" style={{ padding: 'var(--space-lg)' }}>
        <div className="empty-state">
          <div className="empty-state-icon">⚙️</div>
          <div className="empty-state-title">Settings coming soon</div>
          <div className="empty-state-text">System configuration options will be available here</div>
        </div>
      </div>
    </div>
  );
}

// ==================== MAIN APP COMPONENT ====================
function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState('');
  const [view, setView] = useState('live');
  const [logs, setLogs] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [dismissedAlerts, setDismissedAlerts] = useState(new Set());
  const [cameras] = useState(MOCK_CAMERAS);
  const [selectedCamera, setSelectedCamera] = useState(MOCK_CAMERAS[0]);

  // Convert logs to alerts
  const getSeverity = (className) => {
    const critical = ['unknown', 'weapon', 'fire', 'intruder'];
    const warning = ['motion', 'vehicle'];
    const known = ['aum', 'prem', 'auto'];
    const lowerClass = className?.toLowerCase() || '';

    if (critical.some(c => lowerClass.includes(c))) return 'critical';
    if (known.some(k => lowerClass.includes(k))) return 'info';  // Known persons = info level
    if (warning.some(w => lowerClass.includes(w))) return 'warning';
    return 'info';
  };

  const fetchLogs = async () => {
    try {
      const res = await axios.get('http://localhost:3000/api/logs');
      setLogs(res.data);

      // Convert latest logs to alerts (last 10 that aren't dismissed)
      const newAlerts = res.data
        .slice(0, 10)
        .map((log, index) => ({
          id: `${log.timestamp}-${index}`,
          type: getSeverity(log.class),
          message: `${log.class} detected`,
          time: log.timestamp?.split(' ')[1] || 'Unknown',
          camera: log.camera || 'CAM-01',
          timestamp: log.timestamp
        }))
        .filter(alert => !dismissedAlerts.has(alert.id));

      setAlerts(newAlerts);
    } catch (error) {
      console.error('Failed to fetch logs:', error);
    }
  };

  // Auto-refresh alerts every 10 seconds
  useEffect(() => {
    if (isLoggedIn) {
      fetchLogs();
      const interval = setInterval(fetchLogs, 10000);
      return () => clearInterval(interval);
    }
  }, [isLoggedIn, dismissedAlerts]);

  const dismissAlert = (alertId) => {
    setDismissedAlerts(prev => new Set([...prev, alertId]));
    setAlerts(prev => prev.filter(a => a.id !== alertId));
  };

  const handleLogin = (username) => {
    setUser(username);
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser('');
    setView('live');
    setDismissedAlerts(new Set());
  };

  if (!isLoggedIn) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <div className="dashboard">
      <Sidebar
        view={view}
        setView={setView}
        cameras={cameras}
        selectedCamera={selectedCamera}
        setSelectedCamera={setSelectedCamera}
        onLogout={handleLogout}
      />
      <div className="main-content">
        <Header user={user} onLogout={handleLogout} />
        <div className="content-area">
          {view === 'live' && <LiveView camera={selectedCamera} />}
          {view === 'logs' && <LogsView logs={logs} fetchLogs={fetchLogs} />}
          {view === 'settings' && <SettingsView />}
        </div>
      </div>
      <AlertsPanel alerts={alerts} dismissAlert={dismissAlert} />
    </div>
  );
}

export default App;