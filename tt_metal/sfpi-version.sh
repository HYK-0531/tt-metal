# set SFPI release version information

sfpi_version=v6.13.0-binutils
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=17c9f11616e02f2ca7ef85ca9779df89
sfpi_x86_64_Linux_deb_md5=40884be84b0181a6dcb42c797e4d0cf1
sfpi_x86_64_Linux_rpm_md5=4f18d9d9f6e14757ac8a6586f23e1343
