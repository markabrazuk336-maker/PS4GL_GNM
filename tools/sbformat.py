"""sbformat.py — формат готового шейдера PS4 (.sb): разбор и сборка.

    python tools/sbformat.py --check <файл>   # все блоки 0xE891BC71 в файле:
                                              # разобрать -> собрать -> сравнить

Сборка сама вычисляет все размеры, смещения, ссылки на строки и число
в первой инструкции (s_mov_b32 vcc_hi, K). Если результат совпадает с
оригиналом байт в байт, значит, устройство формата понято верно, и тем
же кодом можно собирать свои шейдеры (pack_blob из словаря).

Блок для glShaderBinary (формат 0):
  u32 0xE891BC71, u32 100, u32 1, u32 len(src), src,
  u32 len(program), u32 0xBEEF0400, program
program:
  u64 hash, u8[4] kind, u32 len(gnm), u8[16] нули, gnm, meta
gnm:
  'Shdr', u16 7, u16 2, u8 тип(2=PS,1=VS), u8 hdr_dw, u8 aux, u8 modes, u32 0,
  заголовок hdr_dw слов: [размер|слоты<<24, w5, длина_заголовка, регистры…,
                          счётчики, слоты, семантики]
  код, служебные слова (extra), слоты, хвост (chunk), OrbShdr (28 байт)
meta:
  u32 буферов, констант, элементов, x; u8 входов, выходов, 2 байта; u32 y;
  u32 размер_строк; буферы (24), константы (32), элементы (36),
  входы (16), выходы (16), строки (без повторов, выровнены на 4), хвост 4 байта.
Ссылки на строки — относительные: адрес_поля + значение = адрес строки.
"""
import re
import struct
import sys

BLOB_MAGIC = 0xE891BC71


# ------------------------------------------------------------------ разбор

def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def cstr(b, o):
    e = b.index(b"\0", o)
    return b[o:e].decode("latin1")


def parse_gnm(g):
    assert g[:4] == b"Shdr", "нет Shdr"
    ver_major, ver_minor = struct.unpack_from("<HH", g, 4)
    stype, hdr_dw, aux, modes = g[8:12]
    reserved = u32(g, 12)
    hdr = list(struct.unpack_from("<%dI" % hdr_dw, g, 16))
    code_at = 16 + hdr_dw * 4
    shader_size = hdr[0] & 0x7FFFFF
    srt = (hdr[0] >> 23) & 1
    nslots_hdr = hdr[0] >> 24
    info_at = code_at + shader_size - 28
    assert g[info_at:info_at + 7] == b"OrbShdr", "OrbShdr не там"
    w = u32(g, info_at + 8)
    length = w >> 8
    chunk_base, nslots, flags, info_res = g[info_at + 12:info_at + 16]
    shash, crc = struct.unpack_from("<QI", g, info_at + 16)
    code = g[code_at:code_at + length]
    slots_at = info_at - (chunk_base + nslots) * 4
    extra = g[code_at + length:slots_at]
    slots = g[slots_at:slots_at + nslots * 4]
    chunk = g[slots_at + nslots * 4:info_at]
    k = u32(code, 4)
    assert u32(code, 0) == 0xBEEB03FF and (k + 1) * 8 == info_at - code_at, "vcc_hi не сходится"
    return dict(ver=(ver_major, ver_minor), stype=stype, aux=aux, modes=modes, reserved=reserved,
                hdr_rest=hdr[1:], srt=srt, nslots_hdr=nslots_hdr,
                code=code[8:],                      # без первой инструкции — её вычисляем
                extra=extra, slots=slots, chunk=chunk,
                info=dict(version=g[info_at + 7], low=w & 0xFF, flags=flags, res=info_res, hash=shash, crc=crc))


def parse_meta(m):
    nbuf, ncon, nel, nx = struct.unpack_from("<4I", m, 0)
    nin, nout, c2, c3 = m[16:20]
    y, strsize = struct.unpack_from("<2I", m, 20)
    o = 28
    buffers, consts, elems, ins, outs = [], [], [], [], []
    for _ in range(nbuf):
        f = list(struct.unpack_from("<6I", m, o))
        buffers.append(dict(raw=f[:5], name=cstr(m, o + 20 + f[5])))
        o += 24
    for _ in range(ncon):
        f = list(struct.unpack_from("<8I", m, o))
        consts.append(dict(raw=f[:7], name=cstr(m, o + 28 + f[7])))
        o += 32
    for _ in range(nel):
        f = list(struct.unpack_from("<9I", m, o))
        elems.append(dict(raw=f[:7], name=cstr(m, o + 28 + f[7]), sem=cstr(m, o + 32 + f[8])))
        o += 36
    assert nx == 0, "неизвестная таблица (4-й счётчик) не пуста"
    for lst, n in ((ins, nin), (outs, nout)):
        for _ in range(n):
            f = list(struct.unpack_from("<4I", m, o))
            lst.append(dict(raw=f[:2], name=cstr(m, o + 8 + f[2]), sem=cstr(m, o + 12 + f[3])))
            o += 16
    strings_at = o
    tail = m[strings_at + strsize:]
    return dict(c2=c2, c3=c3, y=y, buffers=buffers, consts=consts, elems=elems, ins=ins, outs=outs,
                tail=tail, _strsize=strsize)


def parse_program(p):
    shash = struct.unpack_from("<Q", p, 0)[0]
    kind = p[8:12]
    gsize = u32(p, 12)
    assert p[16:32] == b"\0" * 16
    return dict(hash=shash, kind=kind, gnm=parse_gnm(p[32:32 + gsize]), meta=parse_meta(p[32 + gsize:]))


def parse_blob(b):
    magic, ver, count, srclen = struct.unpack_from("<4I", b, 0)
    assert magic == BLOB_MAGIC and ver == 100 and count == 1
    src = b[16:16 + srclen]
    psize, tag = struct.unpack_from("<2I", b, 16 + srclen)
    assert tag == 0xBEEF0400
    prog = b[24 + srclen:24 + srclen + psize]
    return dict(src=src, program=parse_program(prog))


# ------------------------------------------------------------------ сборка

def pack_gnm(g):
    body_len = len(g["code"]) + 8 + len(g["extra"]) + len(g["slots"]) + len(g["chunk"])
    assert body_len % 8 == 0, "код+служебное должны делиться на 8"
    k = body_len // 8 - 1
    code = struct.pack("<II", 0xBEEB03FF, k) + g["code"]
    info = g["info"]
    nslots = len(g["slots"]) // 4
    chunk_base = len(g["chunk"]) // 4
    length = len(code)
    orb = b"OrbShdr" + bytes([info["version"]]) + struct.pack("<I", info["low"] | (length << 8)) + \
        bytes([chunk_base, nslots, info["flags"], info["res"]]) + struct.pack("<QI", info["hash"], info["crc"])
    shader = code + g["extra"] + g["slots"] + g["chunk"] + orb
    hdr = [len(shader) | (g["srt"] << 23) | (g["nslots_hdr"] << 24)] + list(g["hdr_rest"])
    head = b"Shdr" + struct.pack("<HH", *g["ver"]) + bytes([g["stype"], len(hdr), g["aux"], g["modes"]]) + \
        struct.pack("<I", g["reserved"]) + struct.pack("<%dI" % len(hdr), *hdr)
    return head + shader


def pack_meta(m):
    # строки — в порядке появления, без повторов
    order = []
    for b in m["buffers"]:
        order.append(b["name"])
    for c in m["consts"]:
        order.append(c["name"])
    for e in m["elems"]:
        order += [e["name"], e["sem"]]
    for a in m["ins"] + m["outs"]:
        order += [a["name"], a["sem"]]
    pos, table = {}, b""
    for s in order:
        if s not in pos:
            pos[s] = len(table)
            table += s.encode("latin1") + b"\0"
    strsize = (len(table) + 3) & ~3
    table = table.ljust(strsize, b"\0")
    n_entries = 24 * len(m["buffers"]) + 32 * len(m["consts"]) + 36 * len(m["elems"]) + \
        16 * (len(m["ins"]) + len(m["outs"]))
    strings_at = 28 + n_entries

    def rel(field_at, s):
        return strings_at + pos[s] - field_at

    out = struct.pack("<4I", len(m["buffers"]), len(m["consts"]), len(m["elems"]), 0) + \
        bytes([len(m["ins"]), len(m["outs"]), m["c2"], m["c3"]]) + struct.pack("<2I", m["y"], strsize)
    for b in m["buffers"]:
        o = len(out)
        out += struct.pack("<5I", *b["raw"]) + struct.pack("<I", rel(o + 20, b["name"]))
    for c in m["consts"]:
        o = len(out)
        out += struct.pack("<7I", *c["raw"]) + struct.pack("<I", rel(o + 28, c["name"]))
    for e in m["elems"]:
        o = len(out)
        out += struct.pack("<7I", *e["raw"]) + struct.pack("<2I", rel(o + 28, e["name"]), rel(o + 32, e["sem"]))
    for a in m["ins"] + m["outs"]:
        o = len(out)
        out += struct.pack("<2I", *a["raw"]) + struct.pack("<2I", rel(o + 8, a["name"]), rel(o + 12, a["sem"]))
    assert len(out) == strings_at
    return out + table + m["tail"]


def pack_program(p):
    gnm = pack_gnm(p["gnm"])
    return struct.pack("<Q", p["hash"]) + p["kind"] + struct.pack("<I", len(gnm)) + b"\0" * 16 + gnm + \
        pack_meta(p["meta"])


def pack_blob(d):
    prog = pack_program(d["program"])
    return struct.pack("<4I", BLOB_MAGIC, 100, 1, len(d["src"])) + d["src"] + \
        struct.pack("<2I", len(prog), 0xBEEF0400) + prog


# ------------------------------------------------------------------ проверка

def check(path):
    data = open(path, "rb").read() + b"\0\0\0\0"     # sbmake пишет блок на 4 байта короче поля длины
    ok = bad = 0
    for mt in re.finditer(struct.pack("<I", BLOB_MAGIC), data):
        o = mt.start()
        srclen = u32(data, o + 12)
        psize = u32(data, o + 16 + srclen)
        orig = data[o:o + 24 + srclen + psize]
        try:
            again = pack_blob(parse_blob(orig))
        except AssertionError as e:
            print("НЕ РАЗОБРАЛ @0x%x: %s" % (o, e))
            bad += 1
            continue
        if again == orig:
            ok += 1
            d = parse_blob(orig)
            g, m = d["program"]["gnm"], d["program"]["meta"]
            print("совпало @0x%x  %s  код %d байт  буферов %d, констант %d, входов %d, выходов %d" % (
                o, "PS" if g["stype"] == 2 else "VS" if g["stype"] == 1 else "тип %d" % g["stype"],
                len(g["code"]) + 8, len(m["buffers"]), len(m["consts"]), len(m["ins"]), len(m["outs"])))
        else:
            bad += 1
            diff = next(i for i in range(min(len(orig), len(again))) if orig[i] != again[i]) \
                if any(orig[i] != again[i] for i in range(min(len(orig), len(again)))) else min(len(orig), len(again))
            print("РАЗНИЦА @0x%x: длина %d/%d, первое отличие в байте %d" % (o, len(orig), len(again), diff))
    print("Итог: %d совпало байт в байт, %d нет" % (ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        sys.exit(check(sys.argv[2]))
    print(__doc__)
