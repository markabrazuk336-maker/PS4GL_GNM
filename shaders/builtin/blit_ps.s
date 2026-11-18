// Blit pixel shader: colour from a texture.
// s[0:7] — texture (T#), s[8:11] — sampler (S#), s12 — PRIM_MASK for m0.
// v0, v1 — barycentric i, j (PERSP_CENTER); parameter 0: x, y = u, v.
s_mov_b32 m0, s12
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
image_sample v[4:7], v[2:3], s[0:7], s[8:11] dmask:0xf
s_waitcnt vmcnt(0)
exp mrt0, v4, v5, v6, v7 done vm
s_endpgm
