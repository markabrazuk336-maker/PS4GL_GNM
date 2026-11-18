uniform sampler2D u_tex;
uniform vec4 u_light;
varying vec2 v_uv;
varying vec3 v_normal;

void main()
{
	vec3 n = normalize(v_normal);
	float d = max(dot(n, u_light.xyz), 0.0);
	vec4 c = texture2D(u_tex, v_uv);
	gl_FragColor = vec4(c.xyz * (0.3 + 0.7 * d), 1.0);
}
