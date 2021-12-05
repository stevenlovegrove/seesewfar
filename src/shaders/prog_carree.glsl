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

varying vec2 v_pos;
uniform vec2 u_dim;
uniform mat3 u_KRbaKinv;
uniform float u_gamma;

uniform sampler2D tex;

void main() {
    vec2 Pa = (v_pos * vec2(0.5,-0.5) + vec2(0.5,0.5)) * u_dim - vec2(0.5);
    vec2 Pb = Project(u_KRbaKinv * Unproject(Pa));
    float x = texture2D(tex,Pix2Tex(Pb,u_dim)).x;
    float I = pow(x, u_gamma);
    gl_FragColor = vec4( vec3(I), 1.0);
}
