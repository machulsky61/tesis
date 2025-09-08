import React from 'react';
import styled from 'styled-components';

const ErrorContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  text-align: center;
`;

const ErrorBox = styled.div`
  background: rgba(255, 255, 255, 0.1);
  padding: 30px;
  border-radius: 12px;
  max-width: 600px;
  margin: 20px;
`;

const ErrorTitle = styled.h1`
  font-size: 2rem;
  margin-bottom: 16px;
  color: #ff6b6b;
`;

const ErrorMessage = styled.p`
  font-size: 1.1rem;
  margin-bottom: 20px;
  line-height: 1.6;
`;

const ErrorDetails = styled.details`
  margin-top: 20px;
  text-align: left;
  
  summary {
    cursor: pointer;
    padding: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    margin-bottom: 10px;
  }
  
  pre {
    background: rgba(0, 0, 0, 0.3);
    padding: 15px;
    border-radius: 6px;
    overflow: auto;
    font-size: 0.9rem;
    white-space: pre-wrap;
  }
`;

const RefreshButton = styled.button`
  background: #4caf50;
  color: white;
  border: none;
  padding: 12px 24px;
  font-size: 1rem;
  border-radius: 6px;
  cursor: pointer;
  margin-top: 20px;
  
  &:hover {
    background: #45a049;
  }
`;

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error);
    console.error('Error info:', errorInfo);
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  handleRefresh = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <ErrorContainer>
          <ErrorBox>
            <ErrorTitle>🚫 Something went wrong</ErrorTitle>
            <ErrorMessage>
              The AI Debate Game encountered an error and couldn't load properly.
              This is likely a configuration or connection issue.
            </ErrorMessage>
            
            <div>
              <strong>Common solutions:</strong>
              <ul style={{ textAlign: 'left', marginTop: '10px' }}>
                <li>Make sure the backend is running on <code>http://localhost:8000</code></li>
                <li>Check that <code>/api/health</code> endpoint responds</li>
                <li>Try refreshing the page</li>
                <li>Clear browser cache (Ctrl+Shift+R)</li>
              </ul>
            </div>

            <RefreshButton onClick={this.handleRefresh}>
              🔄 Refresh Page
            </RefreshButton>

            <ErrorDetails>
              <summary>Technical Details (for debugging)</summary>
              <div>
                <strong>Error:</strong>
                <pre>{this.state.error && this.state.error.toString()}</pre>
                
                <strong>Stack Trace:</strong>
                <pre>{this.state.errorInfo?.componentStack || 'Stack trace not available'}</pre>
                
                <strong>Browser Info:</strong>
                <pre>
                  User Agent: {navigator.userAgent}
                  URL: {window.location.href}
                  Timestamp: {new Date().toISOString()}
                </pre>
              </div>
            </ErrorDetails>
          </ErrorBox>
        </ErrorContainer>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;