"""mkpkg.py — turn a PS4 ELF into an installable fake-signed .pkg (OpenOrbis tools).

    python tools/mkpkg.py app.elf --title "My App" --title-id ABCD00001 -o build/pkg
                          [--version 01.00] [--icon icon0.png] [--add local/file=app0/path] ...

The package is an ordinary game (category gd), which gets about 4.5 GiB of direct memory.
Needs OO_PS4_TOOLCHAIN to point at the OpenOrbis PS4 Toolchain, and .NET for PkgTool.Core.
"""
import argparse
import datetime
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib


def default_icon(path):
    """A 512x512 PNG drawn in code: a shaded cube on a dark background."""
    w = h = 512
    top, left, right = (120, 200, 255), (40, 110, 200), (20, 70, 150)
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            dx, dy = x - 256, y - 256
            u = abs(dx) / 170.0
            c = (18 + y // 16, 22 + y // 14, 38 + y // 10)
            if u <= 1.0:
                ytop = -170 * (1 - u) * 0.58 - 40
                ymid = 170 * (1 - u) * 0.58 - 40
                ybot = ymid + 190
                if ytop <= dy < ymid:
                    c = top
                elif ymid <= dy < ybot:
                    c = left if dx < 0 else right
            rows += bytes(c)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def gp4_text(content_id, files, dirs):
    def tree(prefix, depth):
        kids = sorted(d for d in dirs if d.count("/") == depth and (not prefix or d.startswith(prefix + "/")))
        s = ""
        for d in kids:
            name = d.split("/")[-1]
            inner = tree(d, depth + 1)
            pad = "\t" * (depth + 2)
            s += '%s<dir targ_name="%s"%s\n' % (pad, name, ">\n" + inner + pad + "</dir>" if inner else " />")
        return s
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gp4 = ('<?xml version="1.0"?>\n<psproject xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
           'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" fmt="gp4" version="1000">\n\t<volume>\n'
           '\t\t<volume_type>pkg_ps4_app</volume_type>\n\t\t<volume_id>PS4VOLUME</volume_id>\n'
           '\t\t<volume_ts>%s</volume_ts>\n\t\t<package content_id="%s" passcode="00000000000000000000000000000000"\n'
           '\t\t\tstorage_type="digital50" app_type="full" />\n'
           '\t\t<chunk_info chunk_count="1" scenario_count="1">\n\t\t\t<chunks>\n'
           '\t\t\t\t<chunk id="0" layer_no="0" label="Chunk #0" />\n\t\t\t</chunks>\n'
           '\t\t\t<scenarios default_id="0">\n'
           '\t\t\t\t<scenario id="0" type="sp" initial_chunk_count="1" label="Scenario #0">0</scenario>\n'
           '\t\t\t</scenarios>\n\t\t</chunk_info>\n\t</volume>\n\t<files img_no="0">\n' % (ts, content_id))
    gp4 += "".join('\t\t<file targ_path="%s" orig_path="%s" />\n' % (f, f) for f in files)
    return gp4 + "\t</files>\n\t<rootdir>\n" + tree("", 0) + "\t</rootdir>\n</psproject>\n"


def main():
    ap = argparse.ArgumentParser(description="ELF -> PS4 .pkg")
    ap.add_argument("elf")
    ap.add_argument("--title", required=True)
    ap.add_argument("--title-id", required=True, help="9 characters: 4 letters and 5 digits, e.g. ABCD00001")
    ap.add_argument("--label", help="last part of the content id, up to 16 characters (default: from the title)")
    ap.add_argument("--version", default="01.00", help="application version, NN.NN")
    ap.add_argument("--icon", help="icon0.png, 512x512 (default: a generated one)")
    ap.add_argument("--add", action="append", default=[], metavar="LOCAL=PKG/PATH", help="extra file for /app0")
    ap.add_argument("-o", "--out", default=os.path.join("build", "pkg"))
    a = ap.parse_args()

    tc = os.environ.get("OO_PS4_TOOLCHAIN")
    if not tc or not os.path.isdir(tc):
        raise SystemExit("set OO_PS4_TOOLCHAIN to the OpenOrbis PS4 Toolchain folder")
    win = os.name == "nt"
    bin_dir = os.path.join(tc, "bin", "windows" if win else "linux")
    fself = os.path.join(bin_dir, "create-fself" + (".exe" if win else ""))
    pkgtool = os.path.join(bin_dir, "PkgTool.Core" + (".exe" if win else ""))
    if not re.fullmatch(r"[A-Z]{4}\d{5}", a.title_id):
        raise SystemExit("--title-id must be 4 capital letters and 5 digits")
    label = re.sub(r"[^A-Z0-9]", "", (a.label or a.title).upper())[:16].ljust(16, "0")
    content_id = "IV0000-%s_00-%s" % (a.title_id, label)
    assert len(content_id) == 36, content_id

    def run(args, cwd=None):
        env = dict(os.environ, DOTNET_ROLL_FORWARD="LatestMajor")
        r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit("failed: " + os.path.basename(args[0]))

    stage = os.path.join(a.out, "stage")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "sce_sys", "about"))
    os.makedirs(os.path.join(stage, "sce_module"))

    def copy(src, rel):
        dst = os.path.join(stage, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)

    run([fself, "-in=" + a.elf, "-out=" + os.path.join(a.out, "app.oelf"),
         "-eboot=" + os.path.join(stage, "eboot.bin"), "--paid", "0x3800000000000011"])
    # support files come from the toolchain the user installed; they are not part of this repository
    hw = os.path.join(tc, "samples", "hello_world")
    copy(os.path.join(hw, "sce_module", "libc.prx"), "sce_module/libc.prx")
    copy(os.path.join(hw, "sce_module", "libSceFios2.prx"), "sce_module/libSceFios2.prx")
    copy(os.path.join(hw, "sce_sys", "about", "right.sprx"), "sce_sys/about/right.sprx")
    if a.icon:
        copy(a.icon, "sce_sys/icon0.png")
    else:
        default_icon(os.path.join(stage, "sce_sys", "icon0.png"))
    for item in a.add:
        src, rel = item.split("=", 1)
        copy(src, rel)

    sfo = os.path.join(stage, "sce_sys", "param.sfo")
    run([pkgtool, "sfo_new", sfo])
    for key, typ, size, val in (("APP_TYPE", "Integer", "4", "1"), ("APP_VER", "Utf8", "8", a.version),
                                ("ATTRIBUTE", "Integer", "4", "8388608"), ("CATEGORY", "Utf8", "4", "gd"),
                                ("CONTENT_ID", "Utf8", "48", content_id), ("DOWNLOAD_DATA_SIZE", "Integer", "4", "0"),
                                ("SYSTEM_VER", "Integer", "4", "0"), ("TITLE", "Utf8", "128", a.title),
                                ("TITLE_ID", "Utf8", "12", a.title_id), ("VERSION", "Utf8", "8", a.version)):
        run([pkgtool, "sfo_setentry", sfo, key, "--type", typ, "--maxsize", size, "--value", val])

    # create-gp4 cannot handle subfolders well, so the project file is written here
    files, dirs = [], set()
    for base, _, names in os.walk(stage):
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(base, n), stage).replace("\\", "/")
            files.append(rel)
            parts = rel.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:i]))
    with open(os.path.join(stage, "pkg.gp4"), "w", encoding="utf-8", newline="\n") as f:
        f.write(gp4_text(content_id, files, dirs))
    run([pkgtool, "pkg_build", "pkg.gp4", "."], cwd=stage)
    final = os.path.join(a.out, content_id + ".pkg")
    shutil.move(os.path.join(stage, content_id + ".pkg"), final)
    os.remove(os.path.join(stage, "pkg.gp4"))
    print("package: %s (%.1f MiB)" % (final, os.path.getsize(final) / 1048576))
    return 0


if __name__ == "__main__":
    sys.exit(main())
