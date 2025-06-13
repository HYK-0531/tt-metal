# set SFPI release version information

sfpi_version=v6.12.0-insn-synth
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=6e281cbc77cc6557c82d9c1e45dd921b
sfpi_x86_64_Linux_deb_md5=033a37a057e34939f6f1b22969ad55a1
sfpi_x86_64_Linux_rpm_md5=4e209dd19e9775e910e8e7582885f8d6
