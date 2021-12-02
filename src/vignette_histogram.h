#pragma once

#include <Eigen/Eigen>
#include <random>
#include <pangolin/image/image.h>
#include <stdint.h>

struct PolarHistogram
{
    PolarHistogram(size_t num_bins)
        : sum(Eigen::ArrayXf::Zero(num_bins)),
          num(Eigen::ArrayXi::Zero(num_bins))
    {

    }

    void operator+=(const PolarHistogram& other)
    {
        assert(sum.size() == other.sum.size());
        sum += other.sum;
        num += other.num;
    }

    Eigen::ArrayXf sum;
    Eigen::ArrayXi num;
};

PolarHistogram ComputePolarHistogram(const pangolin::Image<uint16_t>& image, const Eigen::Vector2f& center, size_t num_bins, size_t num_samples)
{
    PolarHistogram hist(num_bins);

    static std::default_random_engine gen;
    std::uniform_int_distribution<size_t> dist_w(0, image.w);
    std::uniform_int_distribution<size_t> dist_h(0, image.h);

    for(size_t i=0; i < num_samples; ++i) {
        const Eigen::Vector2i p{dist_w(gen), dist_h(gen)};
        const uint16_t v = image(p);
        const float rad_pix = (p.cast<float>() - center).norm();
        const int rad_pix_i = std::round(rad_pix);
        if(rad_pix_i < num_bins) {
            hist.sum[rad_pix_i] += v;
            ++hist.num[rad_pix_i];
        }
    }
    return hist;
}
