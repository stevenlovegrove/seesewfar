#pragma once

#include <string>

constexpr char star_shader[] = R"Shader(
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

#expect NUM_CP

varying vec2 v_pos;
uniform vec2 u_dim;
uniform vec3 u_color;
uniform mat3 u_KRbaKinv;
uniform float u_gamma;

uniform float u_spline_interval;
uniform float u_spline_control_points[NUM_CP];
uniform mat3  u_spline_matrix;

uniform sampler2D tex;

vec3 powers3(float u)
{
    return vec3(1,u,u*u);
}

float bspline3(float spline_interval, float spline_control_points[NUM_CP], mat3 spline_matrix, float t)
{
    float tc = t / spline_interval;
    int i = int(tc);
    float u = tc - i;
    vec3 t_pows = powers3(u);
    vec3 coeffs = spline_matrix * t_pows;

    float v = 0;
    if(i+3 <= NUM_CP) {
        for(int j=0; j < 3; ++j) {
            v += coeffs[j] * spline_control_points[i+j];
        }
    }
    return v;
}

vec3 Unproject(vec2 p)
{
    return vec3(p, 1.0);
}

vec2 Project(vec3 p)
{
    return vec2(p.x/p.z, p.y/p.z);
}

vec2 Pix2Tex(vec2 p, vec2 dim)
{
    return (p + vec2(0.5)) / dim;
}

void main() {
    vec2 Pa = (v_pos * vec2(0.5,-0.5) + vec2(0.5,0.5)) * u_dim - vec2(0.5);
    vec2 Pb = Project(u_KRbaKinv * Unproject(Pa));
    float rad = length(Pb - u_dim/2.0);
    float v = bspline3(u_spline_interval, u_spline_control_points, u_spline_matrix, rad);
    float x = texture2D(tex,Pix2Tex(Pb,u_dim)).x / v;
    float I = pow(x, u_gamma);
    gl_FragColor = vec4(I*u_color, 1.0);
}
)Shader";
