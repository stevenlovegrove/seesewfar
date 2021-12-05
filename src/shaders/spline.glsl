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
