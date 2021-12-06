@start vertex
#version 120
attribute vec3 a_position;
varying vec2 v_pos;

void main() {
    gl_Position = vec4(a_position, 1.0);
    v_pos = a_position.xy;
}

@start fragment
#version 120

#include "camera.glsl"
#define M_PI 3.1415926538

varying vec2 v_pos;
uniform vec2 u_dim;
uniform mat3 u_RbaKinv;
uniform vec3 u_color;
uniform float u_gamma;

uniform sampler2D tex;

float sq(float x)
{
    return x*x;
}

void main() {
    vec2 p_a = (v_pos * vec2(0.5,-0.5) + vec2(0.5,0.5)) * u_dim - vec2(0.5);
    vec3 P_b = normalize(u_RbaKinv * Unproject(p_a));
    vec2 lat_lon = vec2(
        atan(P_b.y, P_b.x),
        atan(P_b.z, sqrt(sq(P_b.x) + sq(P_b.y)) )
    );
    vec2 lat_lon_norm = (lat_lon + vec2(M_PI, M_PI/2.0)) / vec2(2*M_PI, M_PI);

    vec3 rgb = texture2D(tex, lat_lon_norm).xyz * u_color;
    vec3 rgb_gamma = vec3( pow(rgb.x, u_gamma), pow(rgb.y, u_gamma), pow(rgb.z, u_gamma));
    gl_FragColor = vec4( rgb_gamma, 1.0);
}
