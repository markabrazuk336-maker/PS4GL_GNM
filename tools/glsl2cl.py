"""glsl2cl.py — шейдер GLSL -> OpenCL C для clang.

    python tools/glsl2cl.py <файл.vert|.frag> [-o файл.cl]
    python tools/glsl2cl.py --check <папка>     # перевести и скомпилировать все

Зачем: на PS4 нет компилятора шейдеров, код видеокарты делаем сами.
GLSL очень похож на OpenCL C, а clang умеет OpenCL C -> LLVM для AMD
(--target=amdgcn--). Поэтому переписываем текст, а не пишем компилятор:
  * uniform  -> поля структуры LMV_U (это буфер констант __GLOBAL_CB__);
  * in/out   -> поля структур LMV_IN / LMV_OUT;
  * texture(samplerN, uv) -> lmv_tex2d(N, uv) и т. п. (реализует «рамка»);
  * discard  -> o->discard = 1; return;
  * vecN -> floatN, vecN(...) -> (floatN)(...), inversesqrt -> rsqrt.
Встроенные функции GLSL — shaders/lmvlib.h.
Получается функция
    void lmv_main(const LMV_U *u, const LMV_IN *in, LMV_OUT *o)
которую потом вызывает «рамка» (входы -> структуры -> выходы), см. tools/sbmake.py.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "shaders", "lmvlib.h")

TYPES = {"vec2": "float2", "vec3": "float3", "vec4": "float4", "ivec2": "int2", "ivec3": "int3",
         "ivec4": "int4", "bvec2": "int2", "bvec3": "int3", "bvec4": "int4", "float": "float",
         "int": "int", "bool": "int", "mat4": "lmv_mat4", "mat3": "lmv_mat3", "mat2": "lmv_mat2"}
SAMPLERS = ("sampler2D", "sampler3D", "samplerCube", "sampler2DShadow")

# Тело шейдера: uniform — в константной памяти (буфер __GLOBAL_CB__),
# входы/выходы — структуры рамки, r — описания текстур (T#/S#).
SIG = "void lmv_main(__constant LMV_U *u, const LMV_IN *in, LMV_OUT *o, const LMV_RES *r)"


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def apply_defines(src):
    """#define без параметров (varying/attribute/gl_FragData и т. п.)."""
    defs = {}
    out = []
    for line in src.split("\n"):
        m = re.match(r"\s*#\s*define\s+(\w+)\s*(.*)$", line)
        if m and "(" not in m.group(1):
            defs[m.group(1)] = m.group(2).strip()
            continue
        if re.match(r"\s*#", line):        # version, extension, pragma — не нужны
            continue
        for k, v in defs.items():
            line = re.sub(r"\b%s\b" % re.escape(k), v, line)
        out.append(line)
    return "\n".join(out)


def _operand_end(s, i):
    """Конец простого операнда с позиции i: (тип)(...), имя(...), имя.поле[..]."""
    while i < len(s) and s[i] in " \t":
        i += 1
    def skip_paren(j, o, c):
        depth = 0
        while j < len(s):
            if s[j] == o:
                depth += 1
            elif s[j] == c:
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return j
    if i < len(s) and s[i] == "(":
        i = skip_paren(i, "(", ")")
        if i < len(s) and s[i] == "(":          # приведение типа: (float4)(...)
            i = skip_paren(i, "(", ")")
    else:
        m = re.match(r"[\w.>\-]+", s[i:])
        i += m.end() if m else 0
        if i < len(s) and s[i] == "(":
            i = skip_paren(i, "(", ")")
    while i < len(s) and s[i] in "[.":
        if s[i] == "[":
            i = skip_paren(i, "[", "]")
        else:
            m = re.match(r"\.\w+", s[i:])
            i += m.end()
    return i


def mul_matrices(s, mats):
    if not mats:
        return s
    pat = re.compile(r"((?:u->|in->)?(?:%s))\s*\*" % "|".join(map(re.escape, mats)))
    while True:
        m = pat.search(s)
        if not m:
            return s
        e = _operand_end(s, m.end())
        s = s[:m.start()] + "lmv_mul(%s, %s)" % (m.group(1), s[m.end():e].strip()) + s[e:]


def translate(src, stage):
    src = apply_defines(strip_comments(src))
    uniforms, ins, outs, samplers = [], [], [], []
    body_lines = []
    decl = re.compile(r"^\s*(?:(?:layout\s*\([^)]*\)|flat|smooth|centroid|invariant|"
                      r"highp|mediump|lowp)\s+)*(uniform|in|out|attribute|varying)\s+"
                      r"(?:(?:highp|mediump|lowp)\s+)?(\w+)\s+(\w+)\s*(\[\s*(\d+)\s*\])?\s*;")
    for line in src.split("\n"):
        if re.match(r"\s*precision\b", line):
            continue
        m = decl.match(line)
        if m:
            q, t, name, _, n = m.groups()
            n = int(n) if n else 0
            if q == "uniform":
                (samplers if t in SAMPLERS else uniforms).append((t, name, n))
            elif q in ("in", "attribute") or (q == "varying" and stage == "frag"):
                ins.append((t, name, n))
            else:
                outs.append((t, name, n))
            continue
        body_lines.append(line)
    body = "\n".join(body_lines)
    # GLSL ES 1.00: выход пиксельного шейдера — встроенный gl_FragColor / gl_FragData
    if stage == "frag" and not outs:
        if re.search(r"\bgl_FragData\b", body):
            outs.append(("vec4", "gl_FragData", 1))
        elif re.search(r"\bgl_FragColor\b", body):
            outs.append(("vec4", "gl_FragColor", 0))

    names_u = {n for _, n, _ in uniforms}
    names_i = {n for _, n, _ in ins}
    names_o = {n for _, n, _ in outs}
    samp_index = {n: i for i, (_, n, _) in enumerate(samplers)}

    def field(t, n, cnt):
        return "\t%s %s%s;\n" % (TYPES.get(t, "float4"), n, "[%d]" % cnt if cnt else "")

    def slot(t, n, cnt):
        # вход/выход рамки: всегда 16 байт на штуку, но тип — настоящий (vec2 = float2 + запас)
        if cnt or t not in ("float", "vec2", "vec3", "int", "bool"):
            return field("vec4", n, cnt)
        pad = {"float": "float lmv_p_%s[3];", "vec2": "float2 lmv_p_%s;", "vec3": "",
               "int": "int lmv_p_%s[3];", "bool": "int lmv_p_%s[3];"}[t]
        return "\t%s %s;%s\n" % (TYPES[t], n, (" " + pad % n) if pad else "")

    # выражения: текстуры, конструкторы, имена
    def tex(m):
        fn, samp = m.group(1), m.group(2)
        unit = samp_index.get(samp, -1)
        kind = {"texture": "lmv_tex2d", "texture2D": "lmv_tex2d", "textureLod": "lmv_tex2d_lod",
                "texture2DLod": "lmv_tex2d_lod", "texture3D": "lmv_tex3d",
                "textureCube": "lmv_texcube"}.get(fn, "lmv_tex2d")
        st = next((t for t, n, _ in samplers if n == samp), "sampler2D")
        if kind == "lmv_tex2d" and st == "sampler3D":
            kind = "lmv_tex3d"
        if kind == "lmv_tex2d" and st == "samplerCube":
            kind = "lmv_texcube"
        return "%s(r, %d, " % (kind, unit)
    body = re.sub(r"\b(texture2DLod|textureLod|texture2D|texture3D|textureCube|texture)\s*\(\s*(\w+)\s*,", tex, body)
    # координаты текстуры обрезаем до нужной длины в самой функции — оставляем как есть
    for g, c in (("vec4", "float4"), ("vec3", "float3"), ("vec2", "float2"),
                 ("ivec4", "int4"), ("ivec3", "int3"), ("ivec2", "int2")):
        body = re.sub(r"\b%s\s*\(" % g, "(%s)(" % c, body)
        body = re.sub(r"\b%s\b" % g, c, body)
    body = re.sub(r"\binversesqrt\b", "rsqrt", body)
    # приведения в стиле GLSL: int(x), float(x), bool(x) -> (int)(x) ...
    body = re.sub(r"(?<![\w)])(int|float)\s*\(", r"(\1)(", body)
    body = re.sub(r"(?<![\w)])bool\s*\(", "(int)(", body)
    body = re.sub(r"\bdiscard\s*;", "{ o->discard = 1; return; }", body)
    body = re.sub(r"\bvoid\s+main\s*\(\s*\)", SIG, body)

    def name_ref(m):
        n = m.group(0)
        if n in names_u:
            return "u->" + n
        if n in names_i:
            return "in->" + n
        if n in names_o:
            return "o->" + n
        if n == "gl_Position":
            return "o->gl_Position"
        if n == "gl_FragCoord":
            return "in->gl_FragCoord"
        return n
    # заменяем имена вне строки «void lmv_main(...)»
    head, sep, rest = body.partition(SIG)
    rest = re.sub(r"(?<![\w.>])[A-Za-z_]\w*", name_ref, rest)
    # «матрица * X» -> lmv_mul(матрица, X): в OpenCL C нет перегрузки операторов
    mats = {n for t, n, _ in uniforms + ins if t.startswith("mat")}
    rest = mul_matrices(rest, mats)
    body = head + sep + rest

    cl = ['#include "lmvlib.h"\n',
          "// буфер констант (__GLOBAL_CB__): все uniform по порядку\n",
          "typedef struct {\n" + "".join(field(t, n, c) if t.startswith("mat") else slot(t, n, c)
                                         for t, n, c in uniforms) + "\tfloat4 lmv_pad;\n} LMV_U;\n",
          "typedef struct {\n" + "".join(slot(t, n, c) for t, n, c in ins) + "\tfloat4 gl_FragCoord;\n} LMV_IN;\n",
          "typedef struct {\n" + "".join(slot(t, n, c) for t, n, c in outs) +
          "\tfloat4 gl_Position;\n\tint discard;\n} LMV_OUT;\n",
          "// текстуры: " + ", ".join("%d=%s %s" % (i, t, n) for i, (t, n, _) in enumerate(samplers)) + "\n",
          body]
    return "".join(cl), dict(uniforms=uniforms, ins=ins, outs=outs, samplers=samplers)


def compile_cl(path_cl, path_ll):
    r = subprocess.run(["clang", "-x", "cl", "-cl-std=CL1.2", "-cl-single-precision-constant",
                        "--target=amdgcn--", "-mcpu=bonaire", "-O2", "-S", "-emit-llvm",
                        "-I", os.path.dirname(LIB), path_cl, "-o", path_ll],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stderr


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        d = sys.argv[2]
        out = os.path.join(d, "cl")
        os.makedirs(out, exist_ok=True)
        ok = bad = 0
        for fn in sorted(os.listdir(d)):
            if not fn.endswith((".vert", ".frag")):
                continue
            stage = "frag" if fn.endswith(".frag") else "vert"
            cl, info = translate(open(os.path.join(d, fn), encoding="utf-8", errors="replace").read(), stage)
            pcl = os.path.join(out, fn + ".cl")
            open(pcl, "w", encoding="utf-8").write(cl)
            rc, err = compile_cl(pcl, pcl[:-3] + ".ll")
            if rc == 0:
                ok += 1
                print("ок   %s  uniform %d, входов %d, выходов %d, текстур %d" % (
                    fn, len(info["uniforms"]), len(info["ins"]), len(info["outs"]), len(info["samplers"])))
            else:
                bad += 1
                first = [l for l in err.splitlines() if "error" in l][:2]
                print("НЕТ  %s\n     %s" % (fn, "\n     ".join(first)))
        print("Итог: %d переведено и скомпилировано clang, %d нет" % (ok, bad))
        return 1 if bad else 0
    src = sys.argv[1]
    stage = "frag" if src.endswith(".frag") else "vert"
    cl, _ = translate(open(src, encoding="utf-8").read(), stage)
    dst = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else src + ".cl"
    open(dst, "w", encoding="utf-8").write(cl)
    print("готово:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
