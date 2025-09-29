#!/bin/bash

# QuantDesk Architecture Diagram Generator
# This script helps generate professional architecture diagrams

echo "🚀 QuantDesk Architecture Diagram Generator"
echo "=============================================="

# Check if mermaid-cli is installed
if ! command -v mmdc &> /dev/null; then
    echo "📦 Installing Mermaid CLI..."
    npm install -g @mermaid-js/mermaid-cli
fi

# Create output directory
mkdir -p diagrams

echo "📊 Generating architecture diagrams..."

# Generate system architecture diagram
echo "  - System Architecture Overview..."
mmdc -i complete-arch.md -o diagrams/system-architecture.png -t dark -b white

# Generate data flow diagram
echo "  - Data Flow Architecture..."
mmdc -i complete-arch.md -o diagrams/data-flow.png -t dark -b white

# Generate trading flow diagram
echo "  - Trading Flow Architecture..."
mmdc -i complete-arch.md -o diagrams/trading-flow.png -t dark -b white

echo "✅ Diagrams generated successfully!"
echo "📁 Output directory: ./diagrams/"
echo ""
echo "🎨 Professional diagram options:"
echo "  1. Open diagram-viewer.html in browser for interactive view"
echo "  2. Use generated PNG files for presentations"
echo "  3. Import Mermaid code into Figma, Draw.io, or Lucidchart"
echo ""
echo "📋 Next steps for investor presentations:"
echo "  - Use diagram-viewer.html for live demos"
echo "  - Export PNG files for pitch decks"
echo "  - Share Mermaid code with technical teams"
echo ""
echo "🔗 Recommended tools for professional presentations:"
echo "  - Figma: https://figma.com (Interactive prototypes)"
echo "  - Draw.io: https://draw.io (Free, web-based)"
echo "  - Lucidchart: https://lucidchart.com (Professional templates)"
echo "  - Mermaid Live: https://mermaid.live (Online editor)"
