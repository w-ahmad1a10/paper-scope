import os
import platform
import socket
import sys

print("host      :", socket.gethostname())
print("python    :", sys.version.split()[0])
print("platform  :", platform.platform())
print("cwd       :", os.getcwd())
print("GENERIC-KERNEL-OK")

out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
with open(os.path.join(out_dir, "proof.txt"), "w") as f:
    f.write("remote execution confirmed on %s\n" % platform.platform())
