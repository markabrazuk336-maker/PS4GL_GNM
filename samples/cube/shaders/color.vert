// Coloured triangle: position and colour come straight from the vertex buffer.
attribute vec3 a_pos;
attribute vec3 a_color;
uniform vec4 u_offset;
varying vec3 v_color;

void main()
{
	v_color = a_color;
	gl_Position = vec4(a_pos.xy + u_offset.xy, a_pos.z, 1.0);
}
