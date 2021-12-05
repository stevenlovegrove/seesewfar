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
