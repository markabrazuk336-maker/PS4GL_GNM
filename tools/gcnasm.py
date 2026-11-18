"""gcnasm.py — GCN assembly text -> machine code, using llvm-mc (gfx700 = PS4 GPU family)."""
import re
import shutil
import subprocess
from pathlib import Path


def assemble(src: Path) -> bytes:
    mc = shutil.which("llvm-mc")
    if not mc:
        raise SystemExit("llvm-mc not found: install LLVM and add it to PATH")
    r = subprocess.run([mc, "-arch=amdgcn", "-mcpu=gfx700", "-show-encoding", str(src)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("%s: %s" % (Path(src).name, r.stderr.strip()))
    out = bytearray()
    for enc in re.findall(r"encoding: \[([^\]]*)\]", r.stdout):
        out += bytes(int(x, 16) for x in enc.split(","))
    return bytes(out)
