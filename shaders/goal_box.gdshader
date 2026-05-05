shader_type spatial;
render_mode unshaded, cull_disabled, depth_prepass_alpha;

uniform vec4 line_color : source_color = vec4(0.32, 0.9, 1.0, 0.95);
uniform vec4 fill_color : source_color = vec4(0.06, 0.16, 0.2, 0.08);
uniform float line_thickness : hint_range(0.01, 0.45, 0.01) = 0.08;
uniform float emission_strength : hint_range(0.0, 4.0, 0.05) = 1.2;

void fragment() {
	vec2 face_uv = abs(UV - vec2(0.5)) * 2.0;
	float edge_band = max(face_uv.x, face_uv.y);
	float edge_mask = smoothstep(1.0 - line_thickness, 1.0, edge_band);
	vec4 color = mix(fill_color, line_color, edge_mask);

	ALBEDO = color.rgb;
	EMISSION = line_color.rgb * edge_mask * emission_strength;
	ALPHA = color.a;
}