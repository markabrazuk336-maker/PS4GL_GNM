"""shadersbC.py — GLSL -> готовый шейдер PS4 (.sb) для glShaderBinary.

    python tools/shadersbC.py <шейдер.vert|.frag> -o <выход.sb>

Цепочка (без компилятора Sony):
  1. tools/glsl2cl.py: GLSL -> OpenCL C, тело lmv_main(u, in, o, r);
  2. clang -x cl --target=amdgcn-- -mcpu=bonaire -> LLVM IR тела;
  3. рамка на LLVM IR (здесь): входы -> структуры, вызов тела,
     текстуры через image_sample, вывод цвета (FP16, compr), discard -> kill;
  4. llvm-link + opt (тело встраивается в рамку) + llc -> код GCN;
  5. tools/sbformat.py: обёртка Shdr/OrbShdr и описание входов и констант.

Договорённость пиксельного шейдера:
  s2:s3  — адрес расширенной таблицы (PtrExtendedUserData);
  s4..11 — T# текстуры 0, s12..15 — S# сэмплера 0;
  расширенная таблица: регистр R >= 16 лежит со смещением (R-16)*4 байт:
     текстура k>=1: T# — регистр 16+(k-1)*12, S# — 24+(k-1)*12;
     буфер констант (V#) — следующий регистр после текстур;
  s16    — служебное для интерполяции (m0), пользовательских SGPR — 16;
  v0,v1  — барицентрические i,j (PERSP_CENTER); вход i — атрибут i.
"""
import os
import re
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glsl2cl      # noqa: E402
import sbformat     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBDIR = os.path.join(ROOT, "shaders")

# служебные слова после кода: драйвер их не читает, важна только длина
EXTRA_NO_RES = b""
CHUNK_NO_RES = struct.pack("<2I", 1, 0xBEEF)
EXTRA_RES = bytes(8)
CHUNK_RES = struct.pack("<3I", 0, 1, 0xBEEF)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise SystemExit("%s:\n%s" % (os.path.basename(cmd[0]), r.stderr[-3000:]))
    return r.stdout


def frame_ps(info):
    """Рамка пиксельного шейдера на LLVM IR."""
    nin = len(info["ins"])
    nout = max(1, len(info["outs"]))
    ntex = len(info["samplers"])
    has_cb = bool(info["uniforms"])
    in_size = 16 * (nin + 1)
    out_size = 16 * nout + 16 + 16          # выходы, gl_Position, discard (+выравнивание)
    disc_off = 16 * nout + 16
    L = ['target triple = "amdgcn--"', "",
         "declare float @llvm.amdgcn.interp.p1(float, i32 immarg, i32 immarg, i32)",
         "declare float @llvm.amdgcn.interp.p2(float, float, i32 immarg, i32 immarg, i32)",
         "declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float)",
         "declare void @llvm.amdgcn.exp.compr.v2f16(i32 immarg, i32 immarg, <2 x half>, <2 x half>, i1 immarg, i1 immarg)",
         "declare void @llvm.amdgcn.kill(i1)",
         "declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)",
         "declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)",
         "declare <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)",
         "declare <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)",
         "declare void @lmv_main(ptr addrspace(4), ptr addrspace(5), ptr addrspace(5), ptr addrspace(5))",
         "",
         # чтение текстуры: описания из LMV_RES (T# по 32 байта, S# по 16 с 128-го)
         "define <4 x float> @lmv_tex2d_(ptr addrspace(5) %r, i32 %unit, <2 x float> %uv) #1 {",
         "  %to = mul i32 %unit, 32",
         "  %tp = getelementptr i8, ptr addrspace(5) %r, i32 %to",
         "  %t = load <8 x i32>, ptr addrspace(5) %tp, align 16",
         "  %so = mul i32 %unit, 16",
         "  %so2 = add i32 %so, 512",
         "  %sp = getelementptr i8, ptr addrspace(5) %r, i32 %so2",
         "  %s = load <4 x i32>, ptr addrspace(5) %sp, align 16",
         "  %u = extractelement <2 x float> %uv, i32 0",
         "  %v = extractelement <2 x float> %uv, i32 1",
         "  %c = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %u, float %v, <8 x i32> %t, <4 x i32> %s, i1 false, i32 0, i32 0)",
         "  ret <4 x float> %c",
         "}",
         "define <4 x float> @lmv_tex2d_lod_(ptr addrspace(5) %r, i32 %unit, <2 x float> %uv, float %lod) #1 {",
         "  %to = mul i32 %unit, 32",
         "  %tp = getelementptr i8, ptr addrspace(5) %r, i32 %to",
         "  %t = load <8 x i32>, ptr addrspace(5) %tp, align 16",
         "  %so = mul i32 %unit, 16",
         "  %so2 = add i32 %so, 512",
         "  %sp = getelementptr i8, ptr addrspace(5) %r, i32 %so2",
         "  %s = load <4 x i32>, ptr addrspace(5) %sp, align 16",
         "  %u = extractelement <2 x float> %uv, i32 0",
         "  %v = extractelement <2 x float> %uv, i32 1",
         "  %c = call <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32 15, float %u, float %v, float %lod, <8 x i32> %t, <4 x i32> %s, i1 false, i32 0, i32 0)",
         "  ret <4 x float> %c",
         "}",
         "define <4 x float> @lmv_tex3d_(ptr addrspace(5) %r, i32 %unit, <3 x float> %uvw) #1 {",
         "  %to = mul i32 %unit, 32",
         "  %tp = getelementptr i8, ptr addrspace(5) %r, i32 %to",
         "  %t = load <8 x i32>, ptr addrspace(5) %tp, align 16",
         "  %so = mul i32 %unit, 16",
         "  %so2 = add i32 %so, 512",
         "  %sp = getelementptr i8, ptr addrspace(5) %r, i32 %so2",
         "  %s = load <4 x i32>, ptr addrspace(5) %sp, align 16",
         "  %u = extractelement <3 x float> %uvw, i32 0",
         "  %v = extractelement <3 x float> %uvw, i32 1",
         "  %w = extractelement <3 x float> %uvw, i32 2",
         "  %c = call <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 15, float %u, float %v, float %w, <8 x i32> %t, <4 x i32> %s, i1 false, i32 0, i32 0)",
         "  ret <4 x float> %c",
         "}",
         "",
         "define amdgpu_ps void @main(i32 inreg %s0, i32 inreg %s1, ptr addrspace(4) inreg %ext, "
         "<8 x i32> inreg %t0, <4 x i32> inreg %smp0, i32 inreg %prim, <2 x float> %ij) #0 {",
         "  %%in = alloca [%d x i8], align 16, addrspace(5)" % in_size,
         "  %%out = alloca [%d x i8], align 16, addrspace(5)" % out_size,
         "  %res = alloca [768 x i8], align 16, addrspace(5)",
         "  %i = extractelement <2 x float> %ij, i32 0",
         "  %j = extractelement <2 x float> %ij, i32 1"]
    # входы: атрибут k, 4 компоненты
    for k in range(nin):
        L.append("  %%a%d = insertelement <4 x float> poison, float 0.0, i32 0" % k)
        prev = "%%a%d" % k
        for c in range(4):
            L.append("  %%p1_%d_%d = call float @llvm.amdgcn.interp.p1(float %%i, i32 %d, i32 %d, i32 %%prim)" % (k, c, c, k))
            L.append("  %%p2_%d_%d = call float @llvm.amdgcn.interp.p2(float %%p1_%d_%d, float %%j, i32 %d, i32 %d, i32 %%prim)"
                     % (k, c, k, c, c, k))
            L.append("  %%v_%d_%d = insertelement <4 x float> %s, float %%p2_%d_%d, i32 %d" % (k, c, prev, k, c, c))
            prev = "%%v_%d_%d" % (k, c)
        L.append("  %%ip%d = getelementptr i8, ptr addrspace(5) %%in, i32 %d" % (k, 16 * k))
        L.append("  store <4 x float> %s, ptr addrspace(5) %%ip%d, align 16" % (prev, k))
    # описания текстур
    L.append("  store <8 x i32> %t0, ptr addrspace(5) %res, align 16")
    L.append("  %rs0 = getelementptr i8, ptr addrspace(5) %res, i32 512")
    L.append("  store <4 x i32> %smp0, ptr addrspace(5) %rs0, align 16")
    for k in range(1, min(ntex, 16)):
        treg, sreg = 16 + (k - 1) * 12, 24 + (k - 1) * 12
        L.append("  %%ept%d = getelementptr i8, ptr addrspace(4) %%ext, i32 %d" % (k, (treg - 16) * 4))
        L.append("  %%tk%d = load <8 x i32>, ptr addrspace(4) %%ept%d, align 4, !invariant.load !0" % (k, k))
        L.append("  %%eps%d = getelementptr i8, ptr addrspace(4) %%ext, i32 %d" % (k, (sreg - 16) * 4))
        L.append("  %%sk%d = load <4 x i32>, ptr addrspace(4) %%eps%d, align 4, !invariant.load !0" % (k, k))
        L.append("  %%rt%d = getelementptr i8, ptr addrspace(5) %%res, i32 %d" % (k, 32 * k))
        L.append("  store <8 x i32> %%tk%d, ptr addrspace(5) %%rt%d, align 16" % (k, k))
        L.append("  %%rss%d = getelementptr i8, ptr addrspace(5) %%res, i32 %d" % (k, 512 + 16 * k))
        L.append("  store <4 x i32> %%sk%d, ptr addrspace(5) %%rss%d, align 16" % (k, k))
    # буфер констант: V# из расширенной таблицы -> адрес
    if has_cb:
        cbreg = cb_reg(ntex)
        L += ["  %%cbp = getelementptr i8, ptr addrspace(4) %%ext, i32 %d" % ((cbreg - 16) * 4),
              "  %cbv = load <4 x i32>, ptr addrspace(4) %cbp, align 4, !invariant.load !0",
              "  %cb0 = extractelement <4 x i32> %cbv, i32 0",
              "  %cb1 = extractelement <4 x i32> %cbv, i32 1",
              "  %cbhi = and i32 %cb1, 65535",
              "  %cblo64 = zext i32 %cb0 to i64",
              "  %cbhi64 = zext i32 %cbhi to i64",
              "  %cbsh = shl i64 %cbhi64, 32",
              "  %cbaddr = or i64 %cbsh, %cblo64",
              "  %u = inttoptr i64 %cbaddr to ptr addrspace(4)"]
    else:
        L.append("  %u = inttoptr i64 0 to ptr addrspace(4)")
    L += ["  %%dp = getelementptr i8, ptr addrspace(5) %%out, i32 %d" % disc_off,
          "  store i32 0, ptr addrspace(5) %dp, align 4",
          "  call void @lmv_main(ptr addrspace(4) %u, ptr addrspace(5) %in, ptr addrspace(5) %out, ptr addrspace(5) %res)",
          "  %disc = load i32, ptr addrspace(5) %dp, align 4",
          "  %keep = icmp eq i32 %disc, 0",
          "  call void @llvm.amdgcn.kill(i1 %keep)",
          "  %col = load <4 x float>, ptr addrspace(5) %out, align 16",
          "  %cr = extractelement <4 x float> %col, i32 0",
          "  %cg = extractelement <4 x float> %col, i32 1",
          "  %cb = extractelement <4 x float> %col, i32 2",
          "  %ca = extractelement <4 x float> %col, i32 3",
          "  %rg = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %cr, float %cg)",
          "  %ba = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %cb, float %ca)",
          "  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %rg, <2 x half> %ba, i1 true, i1 true)",
          "  ret void",
          "}",
          "!0 = !{}",
          # те же настройки процессора, что у тела (clang -mcpu=bonaire = gfx704),
          # иначе LLVM не встраивает тело в рамку
          'attributes #0 = { "target-cpu"="gfx704" "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64" }',
          'attributes #1 = { alwaysinline "target-cpu"="gfx704" "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64" }']
    return "\n".join(L).replace("%%", "%") + "\n"


def cb_reg(ntex):
    return 16 + max(ntex - 1, 0) * 12


# Номер «семантики» varying: 0x0F + номер по порядку объявления.
# Драйвер связывает VS и PS по именам из описания.
def sem_ids(names):
    return [0x0F + k for k in range(len(names))]


MAT_ROWS = {"mat4": 4, "mat3": 3, "mat2": 2}


def cb_meta(info, buffers):
    """Описание __GLOBAL_CB__: каждый uniform — строки по 16 байт."""
    # У каждого uniform есть элемент [тип, смещение в буфере, размер, 0, -1, 0, -1].
    # Типы: float 0x00, vec2 0x01, vec3 0x02, vec4 0x03, int 0x0C, uvec4 0x0F, mat4 0x23.
    # Массив: размер = 16 * N, число элементов N — в четвёртом поле.
    elem_type = {"float": 0x00, "vec2": 0x01, "vec3": 0x02, "vec4": 0x03, "int": 0x0C, "bool": 0x0C,
                 "ivec2": 0x0D, "ivec3": 0x0E, "ivec4": 0x0F, "mat2": 0x21, "mat3": 0x22, "mat4": 0x23}
    nat_size = {"float": 4, "vec2": 8, "vec3": 12, "vec4": 16, "int": 4, "bool": 4, "ivec2": 8, "ivec3": 12,
                "ivec4": 16, "mat2": 32, "mat3": 48, "mat4": 64}
    consts, elems, off = [], [], 0
    for t, n, cnt in info["uniforms"]:
        rows = MAT_ROWS.get(t, 1)
        total = rows * max(cnt, 1)
        start = off
        if rows > 1 or cnt:
            for e in range(total):
                consts.append(dict(raw=[off, len(consts), 0x00160300, 0, 0, 0, 0], name="%s[%d]" % (n, e)))
                off += 16
        else:
            consts.append(dict(raw=[off, len(consts), 0x00160300, 0, 0, 0, 0], name=n))
            off += 16
        size = 16 * total if (cnt or rows > 1) else nat_size.get(t, 16)
        elems.append(dict(raw=[0x03000100 | elem_type.get(t, 0x03), start, size, cnt, 0xFFFFFFFF, 0, 0xFFFFFFFF],
                          name=n, sem="(no_name)"))
    elems.sort(key=lambda e: e["name"].encode())      # элементы по алфавиту (байтово)
    buffers.append(dict(raw=[0, off, 0x006A0416, len(elems), 0], name="__GLOBAL_CB__"))
    return consts, elems


VS_EXTRA = bytes(24)                                 # служебное, драйвер не читает; посл. слово = K
VS_CHUNK = struct.pack("<4I", 7, 0, 1, 0xBEEF)
TYPE_CODE = {"float": 0x00, "vec2": 0x01, "vec3": 0x02, "vec4": 0x03}


def frame_vs(info):
    """Рамка вершинного шейдера: загрузчик вершин, тело, экспорт."""
    nin = len(info["ins"])
    outs = [o for o in info["outs"] if o[1] != "gl_Position"]
    has_cb = bool(info["uniforms"])
    in_size = 16 * (nin + 1)
    out_size = 16 * len(outs) + 16 + 16
    pos_off = 16 * len(outs)
    L = ['target triple = "amdgcn--"', "",
         "declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg)",
         "declare void @lmv_main(ptr addrspace(4), ptr addrspace(5), ptr addrspace(5), ptr addrspace(5))",
         "",
         "define amdgpu_vs void @main(i64 inreg %fetch, ptr addrspace(4) inreg %vbt, i32 inreg %s4, i32 inreg %s5, "
         "i32 inreg %s6, i32 inreg %s7, <4 x i32> inreg %cbv) #0 {",
         "  %%in = alloca [%d x i8], align 16, addrspace(5)" % in_size,
         "  %%out = alloca [%d x i8], align 16, addrspace(5)" % out_size,
         "  %res = alloca [768 x i8], align 16, addrspace(5)"]
    if nin:
        # загрузчик вершин (fetch shader): вызов по адресу в s[0:1],
        # атрибут k приходит в v[4+4k : 7+4k]
        rets = ", ".join(["<4 x float>"] * nin)
        cons = ",".join("={v[%d:%d]}" % (4 + 4 * k, 7 + 4 * k) for k in range(nin))
        clob = ",".join("~{v%d}" % r for r in range(4)) + "," + \
            ",".join("~{s%d}" % r for r in list(range(4, 8)) + list(range(12, 32))) + ",~{vcc},~{scc}"
        rt = "<4 x float>" if nin == 1 else "{ %s }" % rets
        L.append('  %%fa = call %s asm sideeffect "s_swappc_b64 s[0:1], s[0:1]", "%s,{s[0:1]},%s"(i64 %%fetch)'
                 % (rt, cons, clob))
        for k in range(nin):
            if nin == 1:
                L.append("  %%a%d = bitcast <4 x float> %%fa to <4 x float>" % k)
            else:
                L.append("  %%a%d = extractvalue %s %%fa, %d" % (k, rt, k))
            L.append("  %%ip%d = getelementptr i8, ptr addrspace(5) %%in, i32 %d" % (k, 16 * k))
            L.append("  store <4 x float> %%a%d, ptr addrspace(5) %%ip%d, align 16" % (k, k))
    if has_cb:
        L += ["  %cb0 = extractelement <4 x i32> %cbv, i32 0",
              "  %cb1 = extractelement <4 x i32> %cbv, i32 1",
              "  %cbhi = and i32 %cb1, 65535",
              "  %cblo64 = zext i32 %cb0 to i64",
              "  %cbhi64 = zext i32 %cbhi to i64",
              "  %cbsh = shl i64 %cbhi64, 32",
              "  %cbaddr = or i64 %cbsh, %cblo64",
              "  %u = inttoptr i64 %cbaddr to ptr addrspace(4)"]
    else:
        L.append("  %u = inttoptr i64 0 to ptr addrspace(4)")
    L.append("  call void @lmv_main(ptr addrspace(4) %u, ptr addrspace(5) %in, ptr addrspace(5) %out, ptr addrspace(5) %res)")
    for k in range(len(outs)):
        L.append("  %%op%d = getelementptr i8, ptr addrspace(5) %%out, i32 %d" % (k, 16 * k))
        L.append("  %%o%d = load <4 x float>, ptr addrspace(5) %%op%d, align 16" % (k, k))
        comps = ", ".join("float %%o%d_%d" % (k, c) for c in range(4))
        for c in range(4):
            L.append("  %%o%d_%d = extractelement <4 x float> %%o%d, i32 %d" % (k, c, k, c))
        L.append("  call void @llvm.amdgcn.exp.f32(i32 %d, i32 15, %s, i1 false, i1 false)" % (32 + k, comps))
    L += ["  %%pp = getelementptr i8, ptr addrspace(5) %%out, i32 %d" % pos_off,
          "  %pos = load <4 x float>, ptr addrspace(5) %pp, align 16",
          "  %px = extractelement <4 x float> %pos, i32 0",
          "  %py = extractelement <4 x float> %pos, i32 1",
          "  %pz = extractelement <4 x float> %pos, i32 2",
          "  %pw = extractelement <4 x float> %pos, i32 3",
          "  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float %px, float %py, float %pz, float %pw, i1 true, i1 false)",
          "  ret void",
          "}",
          'attributes #0 = { "target-cpu"="gfx704" "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64" }']
    return "\n".join(L).replace("%%", "%") + "\n"


def compile_ps(src_text, work, stage="frag"):
    cl, info = glsl2cl.translate(src_text, stage)
    os.makedirs(work, exist_ok=True)
    pcl = os.path.join(work, "body.cl")
    open(pcl, "w", encoding="utf-8").write(cl)
    run(["clang", "-x", "cl", "-cl-std=CL1.2", "-cl-single-precision-constant", "--target=amdgcn--",
         "-mcpu=bonaire", "-O2", "-S", "-emit-llvm", "-I", LIBDIR, pcl, "-o", os.path.join(work, "body.ll")])
    # тело — всегда встраивать в рамку: отдельный вызов требует стека в scratch-памяти
    body_ll = os.path.join(work, "body.ll")
    txt = open(body_ll, encoding="utf-8").read()
    txt, n = re.subn(r"(define [^\n]*@lmv_main\([^\n]*\)[^\n]*?)(#\d+ \{)", r"\1alwaysinline \2", txt)
    if n != 1:
        raise SystemExit("не нашёл lmv_main в теле")
    open(body_ll, "w", encoding="utf-8").write(txt)
    open(os.path.join(work, "frame.ll"), "w", encoding="utf-8").write(
        frame_ps(info) if stage == "frag" else frame_vs(info))
    run(["llvm-link", "-S", os.path.join(work, "body.ll"), os.path.join(work, "frame.ll"),
         "-o", os.path.join(work, "linked.ll")])
    run(["opt", "-S", "--internalize-public-api-list=main", "-passes=internalize,default<O2>",
         os.path.join(work, "linked.ll"), "-o", os.path.join(work, "opt.ll")])
    asm = run(["llc", "-march=amdgcn", "-mcpu=bonaire", os.path.join(work, "opt.ll"), "-o", "-"])
    open(os.path.join(work, "shader.s"), "w", encoding="utf-8").write(asm)
    run(["llc", "-march=amdgcn", "-mcpu=bonaire", "-filetype=obj", os.path.join(work, "opt.ll"),
         "-o", os.path.join(work, "shader.o")])
    run(["llvm-objcopy", "-O", "binary", "--only-section=.text", os.path.join(work, "shader.o"),
         os.path.join(work, "shader.text")])
    code = open(os.path.join(work, "shader.text"), "rb").read()
    nv = int(re.search(r"NumVgprs:\s*(\d+)", asm).group(1))
    ns = int(re.search(r"NumSgprs:\s*(\d+)", asm).group(1))
    if "scratch" in asm and re.search(r"ScratchSize:\s*([1-9]\d*)", asm):
        raise SystemExit("шейдеру нужна scratch-память — пока не поддержано")
    return code, nv, ns, info


def ps_program(src_text, code, nv, ns, info):
    ntex = len(info["samplers"])
    has_cb = bool(info["uniforms"])
    nin = len(info["ins"])
    # слоты ресурсов: {тип, номер в API, первый SGPR, доп.}
    slots = b""
    if ntex > 1 or has_cb:
        slots += bytes([0x1B, 1, 2, 0])                       # PtrExtendedUserData -> s2
    for k in range(min(ntex, 16)):
        treg = 4 if k == 0 else 16 + (k - 1) * 12
        sreg = 12 if k == 0 else 24 + (k - 1) * 12
        slots += bytes([0x00, k, treg, 0x03]) + bytes([0x01, k, sreg, 0x00])
    if has_cb:
        slots += bytes([0x02, 0, cb_reg(ntex), 0x00])
    nslots = len(slots) // 4
    # код: выравниваем, чтобы код+служебное делились на 8
    extra = EXTRA_RES if nslots else EXTRA_NO_RES
    chunk = CHUNK_RES if nslots else CHUNK_NO_RES
    body = 8 + len(code) + len(extra) + len(slots) + len(chunk)
    if body % 8:
        code += b"\x00\x00\x80\xbf"                           # s_nop 0
    vg = max(1, (nv + 3) // 4) - 1
    sg = max(1, (ns + 2 + 7) // 8) - 1
    rsrc1 = 0x000C0000 | (sg << 6) | vg
    rsrc2 = 16 << 1                                           # 16 пользовательских SGPR
    sems = sem_ids(info["ins"])
    sem_bytes = b"".join(struct.pack("<H", s) for s in sems)
    if len(sem_bytes) % 4:
        sem_bytes += b"\0\0"
    db_ctl = 0x10 | (0x40 if "discard" in src_text else 0)
    regs = [0, 0xFFFFFFFF, rsrc1, rsrc2, 0, 4, 2, 2, nin, 0, db_ctl, 0xF]
    hdr_tail = [nin] + list(struct.unpack("<%dI" % (nslots), slots)) + \
        list(struct.unpack("<%dI" % (len(sem_bytes) // 4), sem_bytes))
    hdr_rest = [0] + regs + hdr_tail
    hdr_rest[1] = (1 + len(hdr_rest)) * 4                     # длина заголовка в байтах
    shash = zlib.crc32(code) | (zlib.crc32(src_text.encode()) << 32)
    gnm = dict(ver=(7, 2), stype=2, aux=0, modes=0, reserved=0, hdr_rest=hdr_rest, srt=0, nslots_hdr=nslots,
               code=code, extra=extra, slots=slots, chunk=chunk,
               info=dict(version=7, low=0x41, flags=0x0C if nslots else 0x08, res=3, hash=shash, crc=zlib.crc32(code)))
    # описание для драйвера
    buffers = [dict(raw=[k, 0, 0x00030502, 0, 0], name=n) for k, (_, n, _) in enumerate(info["samplers"])]
    consts, elems = cb_meta(info, buffers) if has_cb else ([], [])
    ins = [dict(raw=[TYPE_CODE.get(t, 3) | (0x24 << 8) | (k << 24), 0], name=n, sem=n.upper())
           for k, (t, n, _) in enumerate(info["ins"])]
    outs = [dict(raw=[0x03 | (0x38 << 8), 0], name="gl_FragColor", sem="(no_name)")]
    meta = dict(c2=0, c3=0, y=0, buffers=buffers, consts=consts, elems=elems, ins=ins, outs=outs, tail=b"\0\0\0\0")
    return dict(hash=shash, kind=bytes([1, 1, 0, 2]), gnm=gnm, meta=meta)


def vs_program(src_text, code, nv, ns, info):
    nin = len(info["ins"])
    outs = [o for o in info["outs"] if o[1] != "gl_Position"]
    has_cb = bool(info["uniforms"])
    slots = bytes([0x12, 0, 0, 0]) + bytes([0x17, 0, 2, 0x10]) if nin else b""   # загрузчик -> s0, таблица VB -> s2
    if has_cb:
        slots += bytes([0x02, 0, 8, 0])                                           # V# констант -> s8
    nslots = len(slots) // 4
    extra = bytearray(VS_EXTRA)
    chunk = VS_CHUNK
    body = 8 + len(code) + len(extra) + len(slots) + len(chunk)
    if body % 8:
        code += b"\x00\x00\x80\xbf"
        body += 4
    struct.pack_into("<I", extra, len(extra) - 4, body // 8 - 1)                   # последнее слово = K
    vg = max(1, (nv + 3) // 4) - 1
    sg = max(1, (ns + 2 + 7) // 8) - 1
    rsrc1 = 0x000C0000 | (sg << 6) | vg
    user = 12 if has_cb else 4
    rsrc2 = user << 1
    nparam = max(1, len(outs))
    regs = [0, 0xFFFFFFFF, rsrc1, rsrc2, (nparam - 1) << 1, 0x4, 0]
    in_sem = b"".join(bytes([k, 4 + 4 * k, {"vec2": 2, "vec3": 3}.get(t, 4) if t != "float" else 1, 0])
                      for k, (t, n, _) in enumerate(info["ins"]))
    ex_sem = b"".join(bytes([s, k]) for k, s in enumerate(sem_ids(outs)))
    if len(ex_sem) % 4:
        ex_sem += b"\0\0"
    tail = bytes([nin, len(outs), 0, 0]) + slots + in_sem + ex_sem
    hdr_rest = [0] + regs + list(struct.unpack("<%dI" % (len(tail) // 4), tail))
    hdr_rest[1] = (1 + len(hdr_rest)) * 4
    shash = zlib.crc32(code) | (zlib.crc32(src_text.encode()) << 32)
    gnm = dict(ver=(7, 2), stype=1, aux=0, modes=0, reserved=0, hdr_rest=hdr_rest, srt=0, nslots_hdr=nslots,
               code=code, extra=bytes(extra), slots=slots, chunk=chunk,
               info=dict(version=7, low=0x45, flags=0x0C, res=3, hash=shash, crc=zlib.crc32(code)))
    buffers = []
    consts, elems = cb_meta(info, buffers) if has_cb else ([], [])
    ins = [dict(raw=[TYPE_CODE.get(t, 3) | (0x24 << 8) | (k << 24), 0], name=n, sem=n.upper())
           for k, (t, n, _) in enumerate(info["ins"])]
    outl = [dict(raw=[TYPE_CODE.get(t, 3) | (0x24 << 8) | (k << 24), 0], name=n, sem=n.upper())
            for k, (t, n, _) in enumerate(outs)]
    outl.append(dict(raw=[0x03 | (0x2E << 8) | (len(outs) << 24), 0], name="gl_Position", sem="(no_name)"))
    meta = dict(c2=0, c3=0, y=0, buffers=buffers, consts=consts, elems=elems, ins=ins, outs=outl, tail=b"\0\0\0\0")
    return dict(hash=shash, kind=bytes([0, 1, 0, 2]), gnm=gnm, meta=meta)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    src_path = sys.argv[1]
    dst = sys.argv[sys.argv.index("-o") + 1]
    src_text = open(src_path, encoding="utf-8", errors="replace").read()
    work = dst + ".work"
    stage = "vert" if src_path.endswith(".vert") else "frag"
    code, nv, ns, info = compile_ps(src_text, work, stage)
    prog = (ps_program if stage == "frag" else vs_program)(src_text, code, nv, ns, info)
    blob = sbformat.pack_blob(dict(src=src_text.encode("latin1", "replace"), program=prog))
    blob = blob[:-4]                              # поле длины считает и 4-байтный тег
    open(dst, "wb").write(blob)
    again = sbformat.parse_blob(blob + b"\0\0\0\0")
    print("готово: %s  %d байт, код %d байт, VGPR %d, SGPR %d, текстур %d, uniform %d, входов %d"
          % (dst, len(blob), len(code) + 8, nv, ns, len(info["samplers"]), len(info["uniforms"]), len(info["ins"])))
    print("разбор обратно: тип %d, слотов %d" % (again["program"]["gnm"]["stype"], again["program"]["gnm"]["nslots_hdr"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
