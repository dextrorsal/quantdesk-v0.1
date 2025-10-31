// QuantDesk Admin Terminal Layout
import { Outlet } from "react-router";
import { useAuth } from "contexts/AuthContext";
import { LogOut, User } from "react-feather";
import { Button } from "react-bootstrap";
import ThemeToggle from "../components/ThemeToggle";

const RootLayout = () => {
  const { user, logout } = useAuth();

  return (
    <section className="bg-dark">
      <div id="db-wrapper">
        {/* Admin Header */}
        <div className="admin-header" style={{ 
          background: 'var(--bg-secondary, #0a0a0a)', 
          borderBottom: '1px solid var(--primary-500, #3b82f6)',
          padding: '0.5rem 1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div className="d-flex align-items-center">
            <h5 className="mb-0" style={{ color: 'var(--text-primary, #ffffff)', fontFamily: 'Monaco, Consolas, "Courier New", monospace' }}>
              QuantDesk Admin Terminal
            </h5>
          </div>
          
          <div className="d-flex align-items-center">
            <div className="me-3">
              <ThemeToggle />
            </div>
            <div className="d-flex align-items-center me-3">
              <User size={16} className="me-2" color="var(--text-primary, #ffffff)" />
              <span style={{ color: 'var(--text-primary, #ffffff)' }}>{user?.username}</span>
              <span className="badge ms-2" style={{ background: 'var(--primary-500, #3b82f6)' }}>{user?.role}</span>
            </div>
            
            <Button
              variant="outline-danger"
              size="sm"
              onClick={logout}
              className="d-flex align-items-center"
              style={{ borderColor: 'var(--danger-500, #ef4444)', color: 'var(--text-primary, #ffffff)' }}
            >
              <LogOut size={14} className="me-1" />
              Logout
            </Button>
          </div>
        </div>
        
        <div id="page-content">
          <Outlet />
        </div>
      </div>
    </section>
  );
};

export default RootLayout;
