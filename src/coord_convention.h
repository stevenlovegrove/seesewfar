/**********************************************************************************
Copyright (C) Surreal Vision Ltd.
All rights reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
Please contact info@surreal.vision for any questions about the terms and conditions.
***********************************************************************************/

#pragma once

#include <sophus/se3.hpp>

static const Sophus::SO3d RdfOpenGL =
        Sophus::SO3d( (Eigen::Matrix3d() << 1,0,0, 0,-1,0, 0,0,-1).finished() );

static const Sophus::SO3d RdfVision =
        Sophus::SO3d( (Eigen::Matrix3d() << 1,0,0, 0,1,0, 0,0,1).finished() );

static const Sophus::SO3d RdfRobotics =
        Sophus::SO3d( (Eigen::Matrix3d() << 0,0,1, 1,0,0, 0,1,0).finished() );

// Precompute some coordinate transformations.
static const Sophus::SO3d R_vis_gl = RdfVision * RdfOpenGL.inverse();
static const Sophus::SO3d R_vis_rob = RdfVision * RdfRobotics.inverse();
static const Sophus::SO3d R_gl_vis = RdfOpenGL * RdfVision.inverse();
static const Sophus::SO3d R_gl_rob = RdfOpenGL * RdfRobotics.inverse();
static const Sophus::SO3d R_rob_gl = RdfRobotics * RdfOpenGL.inverse();
static const Sophus::SO3d R_rob_vis = RdfRobotics * RdfVision.inverse();

static const std::string rdf_charactors = "RDFLUB";

static const Eigen::Matrix<double, 3, 6> rdf_directions = (Eigen::Matrix<double, 3, 6>() <<
    1, 0, 0, -1, 0, 0,
    0, 1, 0, 0, -1, 0,
    0, 0, 1, 0, 0, -1
    ).finished();

// return T_2b_1b = R_ba * T_2a_1a * R_ab
template<typename Scalar=double>
inline Sophus::SE3<Scalar> ToCoordinateConvention(
        const Sophus::SE3<Scalar>& T_2a_1a,
        const Sophus::SO3<Scalar>& R_ba
        )
{
    Sophus::SE3<Scalar> T_2b_1b;
    T_2b_1b.so3() = R_ba * T_2a_1a.so3() * R_ba.inverse();
    T_2b_1b.translation() = R_ba * T_2a_1a.translation();
    return T_2b_1b;
}

template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 4, 4> ToCoordinateConvention(
    const Eigen::Matrix<Scalar,4,4>& T_2a_1a,
    const Eigen::Matrix<Scalar,3,3>& R_ba
    )
{
    Eigen::Matrix<Scalar, 4, 4> T_2b_1b;
    T_2b_1b.template topLeftCorner<3, 3>() = R_ba * T_2a_1a.template topLeftCorner<3, 3>() * R_ba.transpose();
    T_2b_1b.template topRightCorner<3, 1>() = R_ba * T_2a_1a.template topRightCorner<3, 1>();
    return T_2b_1b;
}

// return T_2b_1b = R_ba * T_2a_1a * R_ab
// R_ba = rdf_b * rdf_a^-1
template<typename Scalar=double>
inline Sophus::SE3<Scalar> ToCoordinateConvention(
        const Sophus::SE3<Scalar>& T_2a_1a,
        const Sophus::SO3<Scalar>& rdf_a,
        const Sophus::SO3<Scalar>& rdf_b
        )
{
    return ToCoordinateConvention<Scalar>(T_2a_1a, rdf_b * rdf_a.inverse());
}

inline Eigen::Matrix3d RdfMatrixFromXyzString(const std::string& str)
{
    if (str.length() != 3) {
        throw std::invalid_argument("Expected XYZ string containing three unique letters from the set R, D, F, L, U, B.");
    }
    else{
        Eigen::Matrix3d XYZ;
        for (int i = 0; i < 3; ++i) {
            int pos = rdf_charactors.find(toupper(str[i]));
            if (pos != (int)rdf_charactors.npos) {
                XYZ.col(i) = rdf_directions.col(pos);
            }
            else{
                throw std::invalid_argument("Expected XYZ string containing three unique letters from the set U,D,L,R,F,B.");
            }
        }
        return XYZ;
    }
}
