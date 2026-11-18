// Clear pixel shader: colour from user SGPRs s0..s3, exported as FP16 (compressed) to mrt0.
v_mov_b32 v2, s1
v_cvt_pkrtz_f16_f32 v0, s0, v2
v_mov_b32 v2, s3
v_cvt_pkrtz_f16_f32 v1, s2, v2
exp mrt0, v0, v0, v1, v1 done compr vm
s_endpgm
