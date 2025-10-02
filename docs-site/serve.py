#!/usr/bin/env python3
"""
QuantDesk Documentation Server

Simple HTTP server to serve the documentation site locally.
This provides a professional documentation interface for the QuantDesk project.

Usage:
    python serve.py [port]
    
Default port: 8080
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path
import urllib.parse
import markdown

class MarkdownHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that can serve markdown files as HTML"""
    
    def do_GET(self):
        """Handle GET requests, converting markdown to HTML if needed"""
        # Parse the URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Check if it's a markdown file
        if path.endswith('.md'):
            self.serve_markdown(path)
        else:
            # Use default handler for other files
            super().do_GET()
    
    def serve_markdown(self, path):
        """Serve a markdown file as HTML"""
        try:
            # Remove leading slash and decode URL
            file_path = urllib.parse.unquote(path[1:])
            
            # Handle relative paths (../docs/...)
            if file_path.startswith('../'):
                # Go up one directory from docs-site to project root
                project_root = Path(__file__).parent.parent
                file_path = project_root / file_path[3:]  # Remove '../'
            else:
                file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                self.send_error(404, f"File not found: {file_path}")
                return
            
            # Read markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Convert to HTML
            html_content = markdown.markdown(markdown_content, extensions=['codehilite', 'fenced_code'])
            
            # Create full HTML page
            full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{os.path.basename(file_path)} - QuantDesk Documentation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            color: #666;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #f8f9fa;
            font-weight: bold;
        }}
        .back-link {{
            margin-bottom: 20px;
        }}
        .back-link a {{
            color: #3498db;
            text-decoration: none;
            font-weight: bold;
        }}
        .back-link a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="/">&larr; Back to Documentation</a>
        </div>
        {html_content}
    </div>
</body>
</html>
"""
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(full_html.encode('utf-8'))
            
        except Exception as e:
            self.send_error(500, f"Error processing markdown: {str(e)}")

def serve_docs(port=8080):
    """Serve the documentation site"""
    
    # Change to docs directory
    docs_dir = Path(__file__).parent
    os.chdir(docs_dir)
    
    # Create server with custom handler
    handler = MarkdownHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🚀 QuantDesk Documentation Server")
            print(f"📚 Serving documentation at: http://localhost:{port}")
            print(f"📁 Serving from: {docs_dir}")
            print(f"🌐 Opening browser...")
            print(f"⏹️  Press Ctrl+C to stop")
            print("-" * 50)
            
            # Open browser (disabled for showcase)
            # webbrowser.open(f'http://localhost:{port}')
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python serve.py {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid port number. Using default port 8080.")
    
    serve_docs(port)
