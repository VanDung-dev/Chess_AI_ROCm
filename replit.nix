{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.rustc
    pkgs.cargo
    pkgs.libjpeg_turbo
    pkgs.libpng
    pkgs.which
  ];
}
