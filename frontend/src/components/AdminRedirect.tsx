import React, { useEffect } from 'react';

const AdminRedirect: React.FC = () => {
  useEffect(() => {
    // For development, redirect to standalone admin dashboard
    if (import.meta.env.DEV) {
      window.location.href = 'http://localhost:5173';
    } else {
      // In production, serve the admin dashboard from the same domain
      window.location.href = '/admin-dashboard/';
    }
  }, []);

  return (
    <div className="min-vh-100 d-flex align-items-center justify-content-center" style={{ background: '#000000' }}>
      <div className="text-center">
        <div className="spinner-border text-primary mb-3" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
        <h4 className="text-white mb-2">Redirecting to Admin Dashboard...</h4>
        <p className="text-muted">Taking you to the QuantDesk Admin Terminal</p>
      </div>
    </div>
  );
};

export default AdminRedirect;
