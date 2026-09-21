#!/bin/sh
# Installs OriginStack for the current user: no root needed.
#   ./install.sh              install (or upgrade in place)
#   ./install.sh --uninstall  remove everything this script installed
set -eu

DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
DEST="$DATA/OriginStack"
BIN="$HOME/.local/bin"
APPS="$DATA/applications"
ICONS="$DATA/icons/hicolor/512x512/apps"

if [ "${1:-}" = "--uninstall" ]; then
    rm -rf "$DEST"
    rm -f "$BIN/originstack-desktop" "$APPS/originstack.desktop" "$ICONS/originstack.png"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" >/dev/null 2>&1 || true
    echo "OriginStack removed."
    exit 0
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$HERE/OriginStack/OriginStack" ] || { echo "Run this from the extracted OriginStack folder." >&2; exit 1; }

rm -rf "$DEST"
mkdir -p "$DEST" "$BIN" "$APPS" "$ICONS"
cp -a "$HERE/OriginStack/." "$DEST/"
cp "$HERE/icon.png" "$ICONS/originstack.png"
ln -sf "$DEST/OriginStack" "$BIN/originstack-desktop"

cat > "$APPS/originstack.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=OriginStack
Comment=Stack and process astrophotography frames
Exec=$DEST/OriginStack
Icon=$ICONS/originstack.png
Terminal=false
Categories=Graphics;Science;Astronomy;
StartupWMClass=OriginStack
EOF
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" >/dev/null 2>&1 || true

echo "Installed to $DEST"
echo "Start it from your application menu, or run: $DEST/OriginStack"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "(Add $BIN to PATH to use the 'originstack-desktop' command.)" ;; esac
