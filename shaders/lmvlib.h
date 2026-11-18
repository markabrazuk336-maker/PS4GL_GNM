// PS4GL_GNM: встроенные функции GLSL для шейдеров, переведённых в OpenCL C
// (clang -x cl --target=amdgcn--). Библиотеки OpenCL у нас нет — свои,
// на встроенных функциях clang. Перегрузка — через overloadable.
#pragma once
#define LMV_OL __attribute__((overloadable, always_inline)) static inline

#define LMV_EACH(F) F(float) F(float2) F(float3) F(float4)

#define LMV_DEF(T) \
LMV_OL T clamp(T x, T a, T b) { return __builtin_elementwise_min(__builtin_elementwise_max(x, a), b); } \
LMV_OL T min(T a, T b) { return __builtin_elementwise_min(a, b); } \
LMV_OL T max(T a, T b) { return __builtin_elementwise_max(a, b); } \
LMV_OL T abs(T a) { return __builtin_elementwise_abs(a); } \
LMV_OL T floor(T a) { return __builtin_elementwise_floor(a); } \
LMV_OL T ceil(T a) { return __builtin_elementwise_ceil(a); } \
LMV_OL T fract(T a) { return a - __builtin_elementwise_floor(a); } \
LMV_OL T sqrt(T a) { return __builtin_elementwise_sqrt(a); } \
LMV_OL T rsqrt(T a) { return (T)(1.0f) / __builtin_elementwise_sqrt(a); } \
LMV_OL T exp2(T a) { return __builtin_elementwise_exp2(a); } \
LMV_OL T log2(T a) { return __builtin_elementwise_log2(a); } \
LMV_OL T pow(T a, T b) { return __builtin_elementwise_exp2(__builtin_elementwise_log2(a) * b); } \
LMV_OL T mix(T a, T b, T t) { return a + (b - a) * t; } \
LMV_OL T step(T e, T x) { return x < e ? (T)(0.0f) : (T)(1.0f); } \
LMV_OL T sign(T x) { return x > (T)(0.0f) ? (T)(1.0f) : (x < (T)(0.0f) ? (T)(-1.0f) : (T)(0.0f)); }
LMV_EACH(LMV_DEF)

// смешанные варианты GLSL: clamp(vec, float, float), mix(vec, vec, float), min/max(vec, float)
#define LMV_MIXED(T) \
LMV_OL T clamp(T x, float a, float b) { return clamp(x, (T)(a), (T)(b)); } \
LMV_OL T mix(T a, T b, float t) { return mix(a, b, (T)(t)); } \
LMV_OL T min(T a, float b) { return min(a, (T)(b)); } \
LMV_OL T max(T a, float b) { return max(a, (T)(b)); } \
LMV_OL T step(float e, T x) { return step((T)(e), x); }
LMV_MIXED(float2) LMV_MIXED(float3) LMV_MIXED(float4)

LMV_OL float dot(float a, float b) { return a * b; }
LMV_OL float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
LMV_OL float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
LMV_OL float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
LMV_OL float length(float2 a) { return sqrt(dot(a, a)); }
LMV_OL float length(float3 a) { return sqrt(dot(a, a)); }
LMV_OL float length(float4 a) { return sqrt(dot(a, a)); }
LMV_OL float2 normalize(float2 a) { return a * rsqrt(dot(a, a)); }
LMV_OL float3 normalize(float3 a) { return a * rsqrt(dot(a, a)); }
LMV_OL float4 normalize(float4 a) { return a * rsqrt(dot(a, a)); }
LMV_OL float3 cross(float3 a, float3 b) { return (float3)(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
LMV_OL float3 reflect(float3 i, float3 n) { return i - 2.0f * dot(n, i) * n; }

// Чтение текстур реализует «рамка» шейдера (tools/shadersbC.py). LMV_RES — описания
// текстур (T#, 8 слов) и сэмплеров (S#, 4 слова), до 16 штук: рамка заполняет их
// из SGPR и расширенной таблицы.
typedef struct { uint8 t[16]; uint4 s[16]; } LMV_RES;
float4 lmv_tex2d_(const LMV_RES *r, int unit, float2 uv);
float4 lmv_tex2d_lod_(const LMV_RES *r, int unit, float2 uv, float lod);
float4 lmv_tex3d_(const LMV_RES *r, int unit, float3 uvw);

// координата любой длины: лишние компоненты отрезаем (входы у нас все float4)
LMV_OL float2 lmv_xy(float2 v) { return v; }
LMV_OL float2 lmv_xy(float3 v) { return v.xy; }
LMV_OL float2 lmv_xy(float4 v) { return v.xy; }
LMV_OL float3 lmv_xyz(float3 v) { return v; }
LMV_OL float3 lmv_xyz(float4 v) { return v.xyz; }
#define lmv_tex2d(r, u, c) lmv_tex2d_(r, u, lmv_xy(c))
#define lmv_tex2d_lod(r, u, c, l) lmv_tex2d_lod_(r, u, lmv_xy(c), l)
#define lmv_tex3d(r, u, c) lmv_tex3d_(r, u, lmv_xyz(c))
#define lmv_texcube(r, u, c) lmv_tex3d_(r, u, lmv_xyz(c)) /* куб пока как 3D — TODO */

// матрицы: в буфере констант — столбцы по 16 байт (mat3 — 3 столбца float4)
typedef struct { float4 c[4]; } lmv_mat4;
typedef struct { float4 c[3]; } lmv_mat3;
typedef struct { float4 c[2]; } lmv_mat2;
LMV_OL float4 lmv_mul(lmv_mat4 m, float4 v) { return m.c[0] * v.x + m.c[1] * v.y + m.c[2] * v.z + m.c[3] * v.w; }
LMV_OL float3 lmv_mul(lmv_mat3 m, float3 v) { return m.c[0].xyz * v.x + m.c[1].xyz * v.y + m.c[2].xyz * v.z; }
LMV_OL float2 lmv_mul(lmv_mat2 m, float2 v) { return m.c[0].xy * v.x + m.c[1].xy * v.y; }
LMV_OL lmv_mat4 lmv_mul(lmv_mat4 a, lmv_mat4 b) {
	lmv_mat4 r;
	for (int i = 0; i < 4; i++) r.c[i] = lmv_mul(a, b.c[i]);
	return r;
}

// Сравнения векторов GLSL (lessThan и т. п.) дают bvec. У нас bvec — вектор
// float из 0.0/1.0 (годится для vec2(lessThan(...))), а any/all/not работают с ним же.
#define LMV_CMP(T) \
LMV_OL T lessThan(T a, T b) { return a < b ? (T)(1.0f) : (T)(0.0f); } \
LMV_OL T lessThanEqual(T a, T b) { return a <= b ? (T)(1.0f) : (T)(0.0f); } \
LMV_OL T greaterThan(T a, T b) { return a > b ? (T)(1.0f) : (T)(0.0f); } \
LMV_OL T greaterThanEqual(T a, T b) { return a >= b ? (T)(1.0f) : (T)(0.0f); } \
LMV_OL T equal(T a, T b) { return a == b ? (T)(1.0f) : (T)(0.0f); } \
LMV_OL T notEqual(T a, T b) { return a != b ? (T)(1.0f) : (T)(0.0f); } \
LMV_OL T not(T a) { return a == (T)(0.0f) ? (T)(1.0f) : (T)(0.0f); }
LMV_CMP(float2) LMV_CMP(float3) LMV_CMP(float4)
LMV_OL int any(float2 a) { return a.x != 0.0f || a.y != 0.0f; }
LMV_OL int any(float3 a) { return any(a.xy) || a.z != 0.0f; }
LMV_OL int any(float4 a) { return any(a.xyz) || a.w != 0.0f; }
LMV_OL int all(float2 a) { return a.x != 0.0f && a.y != 0.0f; }
LMV_OL int all(float3 a) { return all(a.xy) && a.z != 0.0f; }
LMV_OL int all(float4 a) { return all(a.xyz) && a.w != 0.0f; }
