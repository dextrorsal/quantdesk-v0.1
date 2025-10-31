import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

// https://vite.dev/config/
export default defineConfig({
	plugins: [tsconfigPaths(), react()],
	server: {
		port: 5173,
		proxy: {
			'/api': {
				target: 'http://localhost:3002',
				changeOrigin: true,
				secure: false,
			},
		},
	},
});
