#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/home/dex/Desktop/Quantify-1.0.1/web_ui/frontend/src"
DEST_DIR="/home/dex/Desktop/quantdesk/frontend/src/pro/vendor/quantify"

echo "⏳ Importing Quantify UI from: $SRC_DIR"

if [ ! -d "$SRC_DIR" ]; then
  echo "❌ Source not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

# Copy selected folders to keep scope contained
rsync -a --delete \
  "$SRC_DIR/components/" "$DEST_DIR/components/"

rsync -a --delete \
  "$SRC_DIR/pages/" "$DEST_DIR/pages/"

rsync -a --delete \
  "$SRC_DIR/contexts/" "$DEST_DIR/contexts/"

rsync -a --delete \
  "$SRC_DIR/hooks/" "$DEST_DIR/hooks/"

rsync -a --delete \
  "$SRC_DIR/services/" "$DEST_DIR/services/"

# Copy styles into a scoped theme file for Pro route only
mkdir -p "/home/dex/Desktop/quantdesk/frontend/src/pro"
cp -f "$SRC_DIR/index.css" "/home/dex/Desktop/quantdesk/frontend/src/pro/theme.css"

echo "✅ Quantify UI imported to: $DEST_DIR"
echo "Next: wire frontend/src/pro/index.tsx to render vendor components and import theme.css"


