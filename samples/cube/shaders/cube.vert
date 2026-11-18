// Textured cube: a matrix from the program, a simple light per face.
attribute vec3 a_pos;
attribute vec2 a_uv;
attribute vec3 a_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
varying vec2 v_uv;
varying vec3 v_normal;

void main()
{
	v_uv = a_uv;
	vec4 n = u_model * vec4(a_normal, 0.0);
	v_normal = n.xyz;
	gl_Position = u_mvp * vec4(a_pos, 1.0);
}
