#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local}"
BUILD_TYPE="${BUILD_TYPE:-release}"
VERSION="${VERSION:-1.0.0}"

INCLUDEDIR="$PREFIX/include"
LIBDIR="$PREFIX/lib"
PKGDIR="$LIBDIR/pkgconfig"

if [[ -w "$PREFIX" ]] || [[ ! -e "$PREFIX" && -w "$(dirname "$PREFIX")" ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

cargo build --$BUILD_TYPE --lib
$SUDO install -d "$INCLUDEDIR" "$LIBDIR" "$PKGDIR"
$SUDO install -m 0644 include/vlsvrs.h "$INCLUDEDIR/"
OS="$(uname -s)"
MAJOR="${VERSION%%.*}"

if [[ "$OS" == "Linux" ]]; then
  if [[ -f "target/$BUILD_TYPE/libvlsvrs.so" ]]; then
    $SUDO install -m 0755 "target/$BUILD_TYPE/libvlsvrs.so" "$LIBDIR/libvlsvrs.so.$VERSION"
    (
      cd "$LIBDIR"
      $SUDO ln -sf "libvlsvrs.so.$VERSION" "libvlsvrs.so.$MAJOR"
      $SUDO ln -sf "libvlsvrs.so.$MAJOR" "libvlsvrs.so"
    )
  else
    echo "warning: target/$BUILD_TYPE/libvlsvrs.so not found"
  fi

  if [[ -f "target/$BUILD_TYPE/libvlsvrs.a" ]]; then
    $SUDO install -m 0644 "target/$BUILD_TYPE/libvlsvrs.a" "$LIBDIR/"
  fi

  if [[ -n "$SUDO" ]] && command -v ldconfig >/dev/null 2>&1; then
    $SUDO ldconfig
  fi

elif [[ "$OS" == "Darwin" ]]; then
  if [[ -f "target/$BUILD_TYPE/libvlsvrs.dylib" ]]; then
    $SUDO install -m 0755 "target/$BUILD_TYPE/libvlsvrs.dylib" "$LIBDIR/libvlsvrs.$VERSION.dylib"
    (
      cd "$LIBDIR"
      $SUDO ln -sf "libvlsvrs.$VERSION.dylib" "libvlsvrs.$MAJOR.dylib"
      $SUDO ln -sf "libvlsvrs.$MAJOR.dylib" "libvlsvrs.dylib"
    )
    if command -v install_name_tool >/dev/null 2>&1; then
      $SUDO install_name_tool -id "$LIBDIR/libvlsvrs.dylib" "$LIBDIR/libvlsvrs.$VERSION.dylib" || true
    fi
  else
    echo "warning: target/$BUILD_TYPE/libvlsvrs.dylib not found"
  fi

  if [[ -f "target/$BUILD_TYPE/libvlsvrs.a" ]]; then
    $SUDO install -m 0644 "target/$BUILD_TYPE/libvlsvrs.a" "$LIBDIR/"
  fi
else
  echo "Unsupported OS: $OS" >&2
  exit 1
fi

cat <<PC > /tmp/vlsvrs.pc
prefix=$PREFIX
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: vlsvrs
Description: VLSV Rust C bindings
Version: $VERSION
Libs: -L\${libdir} -lvlsvrs
Cflags: -I\${includedir}
PC

$SUDO install -m 0644 /tmp/vlsvrs.pc "$PKGDIR/vlsvrs.pc"
rm -f /tmp/vlsvrs.pc

echo "Installed vlsvrs to $PREFIX"
echo "Header: $INCLUDEDIR/vlsvrs.h"
echo "Libdir: $LIBDIR"
echo "pkg-config: $PKGDIR/vlsvrs.pc"
