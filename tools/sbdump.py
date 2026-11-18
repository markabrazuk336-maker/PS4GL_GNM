"""sbdump.py — разбор готовых шейдеров PS4 (метка OrbShdr) в любом файле.

    python tools/sbdump.py <файл> [--code] [--only N]

Для каждого шейдера печатает:
  * блок ShaderBinaryInfo («OrbShdr»): тип, длина кода, хэш, число слотов;
  * слоты ресурсов (input usage slots): что лежит в каком SGPR —
    константы, текстуры, сэмплеры, вершинные буферы;
  * заголовок «Shdr» перед кодом;
  * имена из таблицы после OrbShdr и исходный GLSL, если он рядом;
  * с --code — код GCN словами по 32 бита (шестнадцатерично).

Формат ShaderBinaryInfo — общеизвестная структура PS4 (по ней работают
открытые эмуляторы; используем как справку по формату, код не берём).
Первая инструкция шейдера — s_mov_b32 vcc_hi, K: блок OrbShdr лежит на
(K + 1) * 8 байт от начала кода.
"""
import os
import re
import struct
import sys

TYPES = {0: "PS (пиксельный)", 1: "VS (вершинный)", 2: "GS", 3: "ES", 4: "HS", 5: "LS", 6: "CS (вычисл.)"}

# Типы слотов ресурсов (usage_type) — что даёт система шейдеру в SGPR.
USAGE = {
    0x00: "ImmResource (T#, текстура/буфер)",
    0x01: "ImmSampler (S#)",
    0x02: "ImmConstBuffer (V#, константы)",
    0x03: "ImmVertexBuffer (V#, вершины)",
    0x04: "ImmRwResource",
    0x05: "ImmAluFloatConst",
    0x06: "ImmAluBool32Const",
    0x07: "ImmGdsCounterRange",
    0x08: "ImmGdsMemoryRange",
    0x09: "ImmGwsBase",
    0x0A: "ImmShaderResourceTable",
    0x0D: "ImmLdsEsGsSize",
    0x12: "SubPtrFetchShader (вершинный загрузчик)",
    0x13: "PtrResourceTable",
    0x14: "PtrInternalResourceTable",
    0x15: "PtrSamplerTable",
    0x16: "PtrConstBufferTable",
    0x17: "PtrVertexBufferTable",
    0x18: "PtrSoBufferTable",
    0x19: "PtrRwResourceTable",
    0x1A: "PtrInternalGlobalTable",
    0x1B: "PtrExtendedUserData",
    0x1C: "PtrIndirectResourceTable",
    0x1D: "PtrIndirectInternalResourceTable",
    0x1E: "PtrIndirectRwResourceTable",
    0x22: "ImmGdsKickRingBufferOffset",
    0x23: "ImmVertexRingBufferOffset",
    0x24: "PtrDispatchDraw",
    0x25: "ImmDispatchDrawInstances",
}


def info_at(d, off):
    """Разобрать ShaderBinaryInfo по смещению метки 'OrbShdr'."""
    version = d[off + 7]
    w = struct.unpack_from("<I", d, off + 8)[0]
    pssl = w & 1
    cached = (w >> 1) & 1
    stype = (w >> 2) & 0xF
    src_type = (w >> 6) & 3
    length = w >> 8
    chunk_base, nslots, flags, _ = d[off + 12:off + 16]
    shash, crc = struct.unpack_from("<QI", d, off + 16)
    return dict(version=version, pssl=pssl, cached=cached, type=stype, src_type=src_type,
                length=length, chunk_base=chunk_base, nslots=nslots, flags=flags, hash=shash, crc=crc)


def find_code_start(d, info_off, length):
    """Начало кода: s_mov_b32 vcc_hi, K, где (K + 1) * 8 = info_off - начало."""
    for start in range(info_off - length - 64, info_off - 7, 4):
        if start < 0:
            continue
        if struct.unpack_from("<I", d, start)[0] != 0xBEEB03FF:
            continue
        k = struct.unpack_from("<I", d, start + 4)[0]
        if start + (k + 1) * 8 == info_off:
            return start
    return None


def slots(d, info_off, inf):
    """Слоты ресурсов: chunk_base слов перед OrbShdr, по 4 байта на слот."""
    out = []
    # таблица слотов стоит на (chunk_base + число слотов) слов раньше метки;
    # последние chunk_base слов перед меткой — служебные (… 0x0000BEEF)
    base = info_off - (inf["chunk_base"] + inf["nslots"]) * 4
    for i in range(inf["nslots"]):
        b = d[base + i * 4: base + i * 4 + 4]
        if len(b) < 4:
            break
        usage, api_slot, start_reg, misc = b
        out.append((usage, api_slot, start_reg, misc))
    return out


def strings_after(d, off, limit=400):
    """Печатные строки после блока OrbShdr (таблица имён) и GLSL-исходник."""
    chunk = d[off: off + limit]
    return [s.decode("latin1") for s in re.findall(rb"[\x20-\x7e]{3,}", chunk)]


def shdr_header(d, code_start):
    """Ближайший 'Shdr' (не OrbShdr) перед кодом."""
    lo = max(0, code_start - 0x200)
    pos = d.rfind(b"Shdr", lo, code_start)
    if pos < 0 or d[pos - 3:pos] == b"Orb":
        return None, []
    words = struct.unpack_from("<%dI" % ((code_start - pos) // 4), d, pos)
    return pos, words


def code_words(d, start, length):
    n = length // 4
    return ["%08x" % w for w in struct.unpack_from("<%dI" % n, d, start)]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    show_code = "--code" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = int(sys.argv[sys.argv.index("--only") + 1])
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    d = open(path, "rb").read()
    found = [m.start() for m in re.finditer(rb"OrbShdr", d)]
    print("%s: шейдеров (OrbShdr) — %d" % (os.path.basename(path), len(found)))
    for n, off in enumerate(found):
        if only is not None and n != only:
            continue
        inf = info_at(d, off)
        cs = find_code_start(d, off, inf["length"])
        print("\n#%d  OrbShdr @0x%x  %s  код %d байт  слотов %d  хэш %016x  pssl=%d src=%d ver=%d"
              % (n, off, TYPES.get(inf["type"], "тип %d" % inf["type"]), inf["length"], inf["nslots"],
                 inf["hash"], inf["pssl"], inf["src_type"], inf["version"]))
        if cs is None:
            print("    начало кода не найдено")
            continue
        print("    код @0x%x" % cs)
        for usage, api, reg, misc in slots(d, off, inf):
            print("    слот: %-40s api %-3d -> s%-3d (доп. 0x%02x)"
                  % (USAGE.get(usage, "тип 0x%02x" % usage), api, reg, misc))
        hpos, words = shdr_header(d, cs)
        if hpos is not None:
            print("    Shdr @0x%x: %s" % (hpos, " ".join("%08x" % w for w in words[1:])))
        names = strings_after(d, off + 0x1c)
        if names:
            print("    строки после: %s" % " | ".join(names[:12]))
        if show_code:
            words = code_words(d, cs, inf["length"])
            for i in range(0, len(words), 8):
                print("      " + " ".join(words[i:i + 8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
