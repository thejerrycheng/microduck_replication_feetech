#!/bin/bash
# Build "Duck Live.app" (the macOS shell, macos/DuckLive.swift) and install it in /Applications.
#   app/build_app.sh            (needs the Xcode command line tools: swiftc)
# The app runs this checkout's server with ~/.venvs/microduck_live; override with DUCK_REPO / DUCK_PYTHON.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="${OUT:-/Applications}"
APP="$OUT/Duck Live.app"
BUILD="$(mktemp -d)"
mkdir -p "$OUT"
rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -target arm64-apple-macos13 "$HERE/macos/DuckLive.swift" -o "$APP/Contents/MacOS/DuckLive" -framework Cocoa -framework WebKit
if [ -f "$HERE/macos/icon.png" ]; then
  IS="$BUILD/icon.iconset"; mkdir -p "$IS"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$HERE/macos/icon.png" --out "$IS/icon_${s}x${s}.png" >/dev/null
    sips -z $((2*s)) $((2*s)) "$HERE/macos/icon.png" --out "$IS/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$IS" -o "$APP/Contents/Resources/AppIcon.icns"
fi
cat > "$APP/Contents/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
 <key>CFBundleName</key><string>Duck Live</string>
 <key>CFBundleDisplayName</key><string>Duck Live</string>
 <key>CFBundleIdentifier</key><string>io.github.thejerrycheng.ducklive</string>
 <key>CFBundleExecutable</key><string>DuckLive</string>
 <key>CFBundleIconFile</key><string>AppIcon</string>
 <key>CFBundlePackageType</key><string>APPL</string>
 <key>CFBundleShortVersionString</key><string>1.0</string>
 <key>LSMinimumSystemVersion</key><string>13.0</string>
 <key>NSHighResolutionCapable</key><true/>
 <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
 <key>DuckRepo</key><string>$REPO</string>
 <key>DuckPython</key><string>${DUCK_PYTHON:-$HOME/.venvs/microduck_live/bin/python}</string>
</dict></plist>
PL
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
rm -rf "$BUILD"
echo "built $APP"
