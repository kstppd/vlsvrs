#!/bin/bash
set -euo pipefail
PREFIX=${PREFIX:-/usr/local}
BUILD_TYPE=${BUILD_TYPE:-release}
INCLUDEDIR="$PREFIX/include"
LIBDIR="$PREFIX/lib"
PKGDIR="$LIBDIR/pkgconfig"

cargo build --$BUILD_TYPE
sudo install -d "$INCLUDEDIR" "$LIBDIR" "$PKGDIR"
# Install header and libs
sudo install -m 0644 include/vlsvrs.h "$INCLUDEDIR/"
if [ -f "target/$BUILD_TYPE/libvlsvrs.so" ]; then
    VERSION="1.0.0"
    sudo install -m 0755 "target/$BUILD_TYPE/libvlsvrs.so" \
        "$LIBDIR/libvlsvrs.so.$VERSION"
    sudo ln -sf "libvlsvrs.so.$VERSION" "$LIBDIR/libvlsvrs.so.1"
    sudo ln -sf "libvlsvrs.so.1" "$LIBDIR/libvlsvrs.so"
fi

if [ -f "target/$BUILD_TYPE/libvlsvrs.a" ]; then
    sudo install -m 0644 "target/$BUILD_TYPE/libvlsvrs.a" "$LIBDIR/"
fi

cat <<PC | sudo tee "$PKGDIR/vlsvrs.pc" >/dev/null
prefix=$PREFIX
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: vlsvrs
Description: VLSV Rust C bindings
Version: 1.0.0
Libs: -L\${libdir} -lvlsvrs
Cflags: -I\${includedir}
PC

if command -v ldconfig >/dev/null 2>&1; then
    sudo ldconfig
fi
echo "Installed vlsvrs to $PREFIX"
