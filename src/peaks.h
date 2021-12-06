#pragma once

#include <Eigen/Eigen>
#include <pangolin/image/image.h>

using namespace pangolin;

struct Peak
{
    Eigen::Vector2f pos;
    float confidence;
};

template <typename T>
void DetectPeaks(
    std::vector<Peak>& peaks,
    const Image<T>& img,
    const float min_confidence
) {
    constexpr int border = 5;
    constexpr float max_upd = 0.807;
    constexpr float min_det = 1e-5;

    for(size_t y = border; y < img.h - border; ++y)
    {
        // starting one element before because we increment at start of loop
        // so that we can use continues mid loop...
        size_t x = border-1;
        const T* p0 = img.RowPtr(y) + border - 1;
        const T* pym = img.RowPtr(y-1) + border - 1;
        const T* pyp = img.RowPtr(y+1) + border - 1;

        const T* pend = img.RowPtr(y) + img.w - border;


        while(p0 != pend)
        {
            ++x;
            ++p0;
            ++pym;
            ++pyp;

            const float dyy = (pyp[0] + pym[0] - 2.0f * p0[0]);
            const float dxx = (p0[1] + p0[-1] - 2.0f * p0[0]);
            if(dxx >= 0 || dyy >= 0) continue; // not maxima

            const float dxy = (pym[-1] - pym[1] - pyp[-1]+ pyp[1]) / 4.0f;
            const float det = (dxx * dyy - dxy * dxy);
            if(fabs(det) <= min_det) continue; // singular

            const float I = p0[0];
            const float dy = (pyp[0] - pym[0]) / 2.0f;
            const float dx = ( p0[1] - p0[-1]) / 2.0f;

            const float confidence = sqrtf(dxx * dxx + dyy * dyy) / I;
            if(confidence < min_confidence) continue; // not strong enough

            const float invdet = 1.f / det;
            const Eigen::Vector2f fp((dxy * dy - dyy * dx) * invdet, (dxy * dx - dxx * dy) * invdet);

            if(fp.lpNorm<Eigen::Infinity>() < max_upd && dxx < 0 && dyy < 0)
            {
                peaks.push_back({(Eigen::Vector2f(x, y) + fp).eval(), confidence});
            }
        }
    }
}

