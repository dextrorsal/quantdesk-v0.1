#!/usr/bin/env python3
"""
Convert all markdown files to HTML for the documentation site
"""

import os
import markdown
from pathlib import Path

def convert_markdown_to_html():
    """Convert all markdown files to HTML"""
    
    # Get project root (parent of docs-site)
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    project_history_dir = project_root / "project_history"
    output_dir = Path(__file__).parent / "html"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Configure markdown
    md = markdown.Markdown(extensions=['codehilite', 'fenced_code', 'tables'])
    
    # HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - QuantDesk Documentation</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <style>
        body {{
            font-family: 'JetBrains Mono', 'Monaco', 'Consolas', monospace;
            line-height: 1.6;
            color: #ffffff;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #000000;
            font-size: 14px;
        }}
        .container {{
            background: #000000;
            padding: 30px;
            border-radius: 8px;
            border: 1px solid #333333;
        }}
        .back-link {{
            margin-bottom: 20px;
        }}
        .back-link a {{
            color: #3b82f6;
            text-decoration: none;
            font-weight: bold;
            transition: color 0.3s ease;
        }}
        .back-link a:hover {{
            color: #60a5fa;
            text-decoration: underline;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #ffffff;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        h1 {{
            border-bottom: 2px solid #333333;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #333333;
            padding-bottom: 8px;
        }}
        code {{
            background: #1a1a1a;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'JetBrains Mono', 'Monaco', 'Consolas', monospace;
            color: #ffffff;
            border: 1px solid #333333;
        }}
        pre {{
            background: #0a0a0a;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #333333;
            margin: 20px 0;
        }}
        pre code {{
            background: none;
            padding: 0;
            color: #ffffff;
            border: none;
        }}
        blockquote {{
            border-left: 4px solid #3b82f6;
            margin: 0;
            padding-left: 20px;
            color: #d9d9d9;
            background: #1a1a1a;
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            border: 1px solid #333333;
        }}
        th, td {{
            border: 1px solid #333333;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #1a1a1a;
            font-weight: bold;
            color: #ffffff;
        }}
        td {{
            color: #d9d9d9;
        }}
        ul, ol {{
            color: #d9d9d9;
        }}
        li {{
            margin-bottom: 8px;
        }}
        a {{
            color: #3b82f6;
            text-decoration: none;
        }}
        a:hover {{
            color: #60a5fa;
            text-decoration: underline;
        }}
        /* Syntax Highlighting Overrides */
        pre[class*="language-"] {{
            background: #0a0a0a !important;
            border: 1px solid #333333 !important;
            border-radius: 8px !important;
            padding: 20px !important;
            margin: 20px 0 !important;
            font-family: 'JetBrains Mono', 'Monaco', 'Consolas', monospace !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
        }}
        .token.comment {{ color: #6a9955 !important; }}
        .token.keyword {{ color: #569cd6 !important; }}
        .token.string {{ color: #ce9178 !important; }}
        .token.number {{ color: #b5cea8 !important; }}
        .token.function {{ color: #dcdcaa !important; }}
        .token.class-name {{ color: #4ec9b0 !important; }}
        .token.operator {{ color: #d4d4d4 !important; }}
        .token.punctuation {{ color: #d4d4d4 !important; }}
        .token.variable {{ color: #9cdcfe !important; }}
        .token.constant {{ color: #4fc1ff !important; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="/">&larr; Back to Documentation</a>
        </div>
        {content}
    </div>
</body>
</html>"""
    
    converted_count = 0
    
    # Convert docs directory
    if docs_dir.exists():
        for md_file in docs_dir.rglob("*.md"):
            try:
                # Read markdown content
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Convert to HTML
                html_content = md.convert(markdown_content)
                
                # Create output path
                relative_path = md_file.relative_to(docs_dir)
                output_path = output_dir / f"docs_{relative_path.with_suffix('.html')}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write HTML file
                title = md_file.stem.replace('_', ' ').title()
                full_html = html_template.format(title=title, content=html_content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_html)
                
                converted_count += 1
                print(f"✅ Converted: {md_file} -> {output_path}")
                
            except Exception as e:
                print(f"❌ Error converting {md_file}: {e}")
    
    # Convert project_history directory
    if project_history_dir.exists():
        for md_file in project_history_dir.rglob("*.md"):
            try:
                # Read markdown content
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Convert to HTML
                html_content = md.convert(markdown_content)
                
                # Create output path
                relative_path = md_file.relative_to(project_history_dir)
                output_path = output_dir / f"history_{relative_path.with_suffix('.html')}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write HTML file
                title = md_file.stem.replace('_', ' ').title()
                full_html = html_template.format(title=title, content=html_content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_html)
                
                converted_count += 1
                print(f"✅ Converted: {md_file} -> {output_path}")
                
            except Exception as e:
                print(f"❌ Error converting {md_file}: {e}")
    
    print(f"\n🎉 Converted {converted_count} markdown files to HTML")
    print(f"📁 HTML files saved to: {output_dir}")

if __name__ == "__main__":
    convert_markdown_to_html()
