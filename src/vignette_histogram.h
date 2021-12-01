#pragma once

#include <Eigen/Eigen>
#include <random>
#include <pangolin/image/image.h>
#include <stdint.h>

Eigen::ArrayXf PolarHistogram(const pangolin::Image<uint16_t>& image, const Eigen::Vector2f& center, size_t num_bins, size_t num_samples)
{
    Eigen::ArrayXf sum = Eigen::ArrayXf::Zero(num_bins);
    Eigen::ArrayXi num = Eigen::ArrayXi::Zero(num_bins);

    static std::default_random_engine gen;
    std::uniform_int_distribution<size_t> dist_w(0, image.w);
    std::uniform_int_distribution<size_t> dist_h(0, image.h);

    for(size_t i=0; i < num_samples; ++i) {
        const Eigen::Vector2i p{dist_w(gen), dist_h(gen)};
        const uint16_t v = image(p);
        const float rad_pix = (p.cast<float>() - center).norm();
        const int rad_pix_i = std::round(rad_pix);
        if(rad_pix_i < num_bins) {
            sum[rad_pix_i] += v;
            ++num[rad_pix_i];
        }
    }
    return sum / num.cast<float>();
}
