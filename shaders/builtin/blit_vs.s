// Blit / clear vertex shader: vertices come straight from a buffer.
// s[0:3] — buffer descriptor (V#) from SPI_SHADER_USER_DATA_VS_0..3.
// v0 — vertex number. A vertex is 32 bytes: position xyzw, then parameter xyzw.
buffer_load_format_xyzw v[1:4], v0, s[0:3], 0 idxen
buffer_load_format_xyzw v[5:8], v0, s[0:3], 0 idxen offset:16
s_waitcnt vmcnt(0)
exp param0, v5, v6, v7, v8
exp pos0, v1, v2, v3, v4 done
s_endpgm
