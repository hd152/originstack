OriginStack for Linux
=====================

Install for your user (no root needed):

    ./install.sh

That copies the app to ~/.local/share/OriginStack, adds an entry to your application menu and a
"originstack-desktop" command in ~/.local/bin. To remove it: ./install.sh --uninstall

Or just run it in place: ./OriginStack/OriginStack

What you need
-------------
* A 64-bit Linux desktop with a graphical session (X11 or Wayland via XWayland).
* A glibc at least as new as the one it was built with: Ubuntu 22.04 or newer, Debian 12 or newer,
  Fedora 36 or newer, or anything of the same age.
* The Tk libraries are bundled; you do not need python3-tk.
* Free disk space for temporary files: roughly 100 MB per frame while a run is going.

Where things go
---------------
Logs: ~/.local/state/OriginStack/logs/ (or $XDG_STATE_HOME).

Not tested on macOS. This build has had less real-world use than the Windows installer: if
something does not work, please open an issue at https://github.com/hd152/originstack/issues and
attach the log.

Project: https://github.com/hd152/originstack   Website: https://hd152.github.io/originstack/
Licence: MIT
