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
    varying vec2 v_pos;
    uniform vec2 u_dim;
    uniform vec3 u_color;
    uniform mat3 u_KRbaKinv;
    uniform float u_gamma;
    uniform float u_vig_scale;
    uniform sampler2D tex;

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
        float x = texture2D(tex,Pix2Tex(Pb,u_dim)).x / 65536.0;
    //    float theta = u_vig_scale*length(v_pos);
    //    float cth = cos(theta);
    //    float A = 1.0 / (cth*cth);
        float I = pow(x, u_gamma);
        gl_FragColor = vec4(I*u_color, 1.0);
    }
)Shader";
