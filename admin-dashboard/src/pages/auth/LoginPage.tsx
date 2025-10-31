// QuantDesk Admin Login Page
import React, { useState, useCallback, useRef } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert } from 'react-bootstrap';
import { Lock, Eye, EyeOff, Chrome, GitHub } from 'react-feather';

interface LoginPageProps {
  onLogin: (token: string) => void;
}

const LoginPage: React.FC<LoginPageProps> = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    twoFactorCode: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showTwoFactor, setShowTwoFactor] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [retryCount, setRetryCount] = useState(0);
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Clear any existing debounce timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    // Debounce the submit to prevent rapid-fire requests
    debounceTimeoutRef.current = setTimeout(async () => {
      setIsLoading(true);
      setError('');

      const maxRetries = 3;
      let currentRetry = retryCount;

      while (currentRetry <= maxRetries) {
        try {
          const response = await fetch('/api/admin/login', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              username: formData.username,
              password: formData.password,
              twoFactorCode: formData.twoFactorCode || undefined
            })
          });

          if (response.ok) {
            const data = await response.json();
            setRetryCount(0); // Reset retry count on success
            onLogin(data.token);
            return;
          } else if (response.status === 429) {
            // Handle rate limiting with exponential backoff
            const errorData = await response.json();
            const retryAfter = errorData.retryAfter || Math.pow(2, currentRetry);
            
            if (currentRetry < maxRetries) {
              setError(`Too many requests. Retrying in ${retryAfter} seconds... (attempt ${currentRetry + 1}/${maxRetries + 1})`);
              await delay(retryAfter * 1000);
              currentRetry++;
              continue;
            } else {
              setError('Too many login attempts. Please wait a few minutes before trying again.');
              setRetryCount(currentRetry);
              return;
            }
          } else {
            const errorData = await response.json();
            if (errorData.requiresTwoFactor) {
              setShowTwoFactor(true);
              setError('Please enter your 2FA code');
            } else {
              setError(errorData.message || 'Login failed');
            }
            setRetryCount(0); // Reset retry count on non-rate-limit errors
            return;
          }
        } catch (err) {
          if (currentRetry < maxRetries) {
            setError(`Network error. Retrying... (attempt ${currentRetry + 1}/${maxRetries + 1})`);
            await delay(Math.pow(2, currentRetry) * 1000); // Exponential backoff
            currentRetry++;
            continue;
          } else {
            setError('Network error. Please check your connection and try again.');
            setRetryCount(currentRetry);
            return;
          }
        }
      }
    }, 500); // 500ms debounce delay

    setIsLoading(false);
  }, [formData, retryCount, onLogin]);

  // Cleanup timeout on unmount
  React.useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="min-vh-100 d-flex align-items-center" style={{ background: '#000000' }}>
      <Container>
        <Row className="justify-content-center">
          <Col md={6} lg={4}>
            <Card className="terminal-card" style={{ border: '1px solid #3b82f6' }}>
              <Card.Header className="text-center" style={{ background: '#1a1a1a', borderBottom: '1px solid #3b82f6' }}>
                <h4 className="mb-0 text-white">
                  <Lock className="me-2" />
                  QuantDesk Admin
                </h4>
                <small className="text-muted">Secure Admin Access</small>
              </Card.Header>
              
              <Card.Body className="p-4">
                {error && (
                  <Alert variant="danger" className="terminal-alert">
                    {error}
                  </Alert>
                )}

                <Form onSubmit={handleSubmit}>
                  <Form.Group className="mb-3">
                    <Form.Label className="text-white">Username</Form.Label>
                    <Form.Control
                      type="text"
                      name="username"
                      value={formData.username}
                      onChange={handleInputChange}
                      required
                      className="terminal-input"
                      style={{ background: '#1a1a1a', border: '1px solid #333', color: '#fff' }}
                    />
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label className="text-white">Password</Form.Label>
                    <div className="position-relative">
                      <Form.Control
                        type={showPassword ? 'text' : 'password'}
                        name="password"
                        value={formData.password}
                        onChange={handleInputChange}
                        required
                        className="terminal-input"
                        style={{ background: '#1a1a1a', border: '1px solid #333', color: '#fff', paddingRight: '50px' }}
                      />
                      <Button
                        type="button"
                        variant="link"
                        className="position-absolute end-0 top-50 translate-middle-y"
                        onClick={() => setShowPassword(!showPassword)}
                        style={{ color: '#3b82f6', padding: '0 10px' }}
                      >
                        {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                      </Button>
                    </div>
                  </Form.Group>

                  {showTwoFactor && (
                    <Form.Group className="mb-3">
                      <Form.Label className="text-white">2FA Code</Form.Label>
                      <Form.Control
                        type="text"
                        name="twoFactorCode"
                        value={formData.twoFactorCode}
                        onChange={handleInputChange}
                        placeholder="Enter 6-digit code"
                        maxLength={6}
                        className="terminal-input"
                        style={{ background: '#1a1a1a', border: '1px solid #333', color: '#fff' }}
                      />
                    </Form.Group>
                  )}

                  <Button
                    type="submit"
                    disabled={isLoading}
                    className="w-100 terminal-button mb-3"
                    style={{ 
                      background: '#3b82f6', 
                      border: '1px solid #3b82f6',
                      color: '#000',
                      fontWeight: '600'
                    }}
                  >
                    {isLoading ? 'Authenticating...' : 'Login'}
                  </Button>
                </Form>

                <hr className="my-4" style={{ borderColor: '#333' }} />
                
                <div className="text-center mb-3">
                  <small className="text-muted">Or login with OAuth</small>
                </div>

                <div className="d-grid gap-2">
                  <Button
                    variant="outline-light"
                    className="d-flex align-items-center justify-content-center"
                    onClick={() => {
                      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002';
                      window.location.href = `${apiUrl}/api/admin/auth/google`;
                    }}
                    style={{ 
                      border: '1px solid #333',
                      color: '#fff',
                      background: 'transparent'
                    }}
                  >
                    <Chrome size={18} className="me-2" />
                    Login with Google
                  </Button>
                  
                  <Button
                    variant="outline-light"
                    className="d-flex align-items-center justify-content-center"
                    onClick={() => {
                      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002';
                      window.location.href = `${apiUrl}/api/admin/auth/github`;
                    }}
                    style={{ 
                      border: '1px solid #333',
                      color: '#fff',
                      background: 'transparent'
                    }}
                  >
                    <GitHub size={18} className="me-2" />
                    Login with GitHub
                  </Button>
                </div>

                <div className="text-center mt-3">
                  <small className="text-muted">
                    🔒 Secure admin access required
                  </small>
                </div>
              </Card.Body>
            </Card>

            <div className="text-center mt-3">
              <small className="text-muted">
                QuantDesk Admin Terminal v1.0.0
              </small>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default LoginPage;
