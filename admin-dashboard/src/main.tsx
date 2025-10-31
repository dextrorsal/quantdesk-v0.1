import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/theme.scss";
import App from "./App.tsx";
import { ThemeProvider } from "./contexts/ThemeContext";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>
);
