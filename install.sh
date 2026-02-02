#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-/usr/local}"      
BUILD_TYPE="${BUILD_TYPE:-release}" 
VERSION="${VERSION:-1.0.0}"         

INCLUDEDIR="$PREFIX/include"
LIBDIR="$PREFIX/lib"
PKGDIR="$LIBDIR/pkgconfig"

cargo build --$BUILD_TYPE --features=vlsv_ptr,vlsv_view,vlsv_dump,zfp
sudo install -d "$INCLUDEDIR" "$LIBDIR" "$PKGDIR"
sudo install -m 0644 include/vlsvrs.h "$INCLUDEDIR/"
OS="$(uname -s)"

# Install libraries
if [[ "$OS" == "Linux" ]]; then
  if [[ -f "target/$BUILD_TYPE/libvlsvrs.so" ]]; then
    sudo install -m 0755 "target/$BUILD_TYPE/libvlsvrs.so" "$LIBDIR/libvlsvrs.so.$VERSION"
    MAJOR="${VERSION%%.*}"
    (cd "$LIBDIR" && sudo ln -sf "libvlsvrs.so.$VERSION" "libvlsvrs.so.$MAJOR")
    (cd "$LIBDIR" && sudo ln -sf "libvlsvrs.so.$MAJOR" "libvlsvrs.so")
  else
    echo "warning: target/$BUILD_TYPE/libvlsvrs.so not found"
  fi

  if [[ -f "target/$BUILD_TYPE/libvlsvrs.a" ]]; then
    sudo install -m 0644 "target/$BUILD_TYPE/libvlsvrs.a" "$LIBDIR/"
  fi

  if command -v ldconfig >/dev/null 2>&1; then
    sudo ldconfig
  fi

elif [[ "$OS" == "Darwin" ]]; then
  if [[ -f "target/$BUILD_TYPE/libvlsvrs.dylib" ]]; then
    MAJOR="${VERSION%%.*}"
    sudo install -m 0755 "target/$BUILD_TYPE/libvlsvrs.dylib" "$LIBDIR/libvlsvrs.$VERSION.dylib"
    (cd "$LIBDIR" && sudo ln -sf "libvlsvrs.$VERSION.dylib" "libvlsvrs.$MAJOR.dylib")
    (cd "$LIBDIR" && sudo ln -sf "libvlsvrs.$MAJOR.dylib" "libvlsvrs.dylib")
    if command -v install_name_tool >/dev/null 2>&1; then
      sudo install_name_tool -id "$LIBDIR/libvlsvrs.dylib" "$LIBDIR/libvlsvrs.$VERSION.dylib" || true
    fi
  else
    echo "warning: target/$BUILD_TYPE/libvlsvrs.dylib not found"
  fi

  if [[ -f "target/$BUILD_TYPE/libvlsvrs.a" ]]; then
    sudo install -m 0644 "target/$BUILD_TYPE/libvlsvrs.a" "$LIBDIR/"
  fi
else
  echo "Unsupported OS: $OS" >&2
  exit 1
fi

cat <<PC | sudo tee "$PKGDIR/vlsvrs.pc" >/dev/null
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

echo "   Installed vlsvrs to $PREFIX"
echo "   Header: $INCLUDEDIR/vlsvrs.h"
echo "   Libdir: $LIBDIR"
echo "   pkg-config: $PKGDIR/vlsvrs.pc"
