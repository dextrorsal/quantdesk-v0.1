// QuantDesk Admin Terminal
import { createBrowserRouter, RouterProvider } from "react-router-dom";

// Import QuantDesk-specific components
import RootLayout from "layouts/RootLayout";
import AdminDashboard from "pages/admin/AdminDashboard";
import ProtectedRoute from "components/auth/ProtectedRoute";
import { AuthProvider } from "contexts/AuthContext";

const App = () => {
  const router = createBrowserRouter([
    {
      id: "root",
      path: "/",
      Component: RootLayout,
      children: [
        {
          id: "admin",
          path: "/",
          element: (
            <ProtectedRoute>
              <AdminDashboard />
            </ProtectedRoute>
          ),
        },
      ],
    },
  ]);
  
  return (
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  );
};

export default App;
