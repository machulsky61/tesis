import React, { useEffect, useState } from 'react';
import styled from 'styled-components';

const DebugContainer = styled.div`
  position: fixed;
  top: 10px;
  right: 10px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 10px;
  border-radius: 6px;
  font-size: 12px;
  max-width: 300px;
  z-index: 1000;
  font-family: monospace;
`;

const StatusItem = styled.div`
  margin: 4px 0;
  padding: 2px 0;
  
  &.success { color: #4caf50; }
  &.error { color: #f44336; }
  &.warning { color: #ff9800; }
`;

const ToggleButton = styled.button`
  position: fixed;
  top: 10px;
  right: 10px;
  background: #333;
  color: white;
  border: none;
  padding: 8px 12px;
  border-radius: 4px;
  cursor: pointer;
  z-index: 1001;
  
  &:hover {
    background: #555;
  }
`;

const DebugInfo = () => {
  const [visible, setVisible] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [apiUrl, setApiUrl] = useState('unknown');

  useEffect(() => {
    // Check API configuration
    const apiBaseUrl = process.env.REACT_APP_API_URL || '/api';
    setApiUrl(apiBaseUrl);

    // Test backend connection
    const checkBackend = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/health`, {
          method: 'GET',
          headers: { 'Accept': 'application/json' }
        });
        
        if (response.ok) {
          setBackendStatus('connected');
        } else {
          setBackendStatus(`error: ${response.status}`);
        }
      } catch (error) {
        setBackendStatus(`error: ${error.message}`);
      }
    };

    checkBackend();
  }, []);

  if (!visible) {
    return (
      <ToggleButton onClick={() => setVisible(true)}>
        🐛 Debug
      </ToggleButton>
    );
  }

  return (
    <DebugContainer>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <strong>Debug Info</strong>
        <button 
          onClick={() => setVisible(false)}
          style={{ background: 'transparent', border: 'none', color: 'white', cursor: 'pointer' }}
        >
          ✕
        </button>
      </div>
      
      <StatusItem className={backendStatus === 'connected' ? 'success' : 'error'}>
        Backend: {backendStatus}
      </StatusItem>
      
      <StatusItem>
        API URL: {apiUrl}
      </StatusItem>
      
      <StatusItem>
        Environment: {process.env.NODE_ENV || 'development'}
      </StatusItem>
      
      <StatusItem>
        React Version: {React.version}
      </StatusItem>
      
      <StatusItem>
        User Agent: {navigator.userAgent.substring(0, 50)}...
      </StatusItem>
      
      <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.7 }}>
        If backend status is "error", make sure:
        <br />• Backend is running on localhost:8000
        <br />• CORS is configured correctly
        <br />• No firewall blocking the connection
      </div>
    </DebugContainer>
  );
};

export default DebugInfo;