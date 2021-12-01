#pragma once

#include <Eigen/Eigen>

double pixel_focal_length_from_mm(double focal_length_mm, const Eigen::Vector2d& sensor_dim_mm, const Eigen::Vector2d& image_dim_pix)
{
    const Eigen::Vector2d pix_per_mm = image_dim_pix.array() / sensor_dim_mm.array();
    return focal_length_mm * pix_per_mm[0];
}
