#!/bin/bash
# Build .venv/Kardia6L.app: a copy of the framework Python's Python.app bundle with
# NSBluetoothAlwaysUsageDescription added, ad-hoc signed. Without this key macOS
# kills any CoreBluetooth-using CLI python with SIGABRT (TCC). This avoids
# modifying /Library/Frameworks. Requires python.org (framework) Python.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HERE/.venv/bin/python"
[ -x "$PY" ] || { echo "create .venv first (python3 -m venv .venv)"; exit 1; }
BASE="$($PY -c 'import sys; print(sys.base_prefix)')"
SRC="$BASE/Resources/Python.app"
[ -d "$SRC" ] || { echo "no Python.app at $SRC — not a framework Python? see README (macOS Bluetooth permission)"; exit 1; }
DST="$HERE/.venv/Kardia6L.app"
rm -rf "$DST"; cp -R "$SRC" "$DST"
P="$DST/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :NSBluetoothAlwaysUsageDescription string 'kardia6l uses Bluetooth to talk to your KardiaMobile 6L'" "$P" 2>/dev/null || \
/usr/libexec/PlistBuddy -c "Set :NSBluetoothAlwaysUsageDescription 'kardia6l uses Bluetooth to talk to your KardiaMobile 6L'" "$P"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier org.kardia6l.python" "$P"
/usr/libexec/PlistBuddy -c "Set :CFBundleName Kardia6L" "$P"
codesign --force --deep --sign - "$DST"
echo "ok: use bin/btpython instead of python for anything that touches Bluetooth"
