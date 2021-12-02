#pragma once

#include <cstddef>
#include <concepts>
#include <Eigen/Eigen>
#include <iostream>

constexpr size_t factorial(size_t x)
{
    if(x==0) {
        return 1;
    }else{
        return x * factorial(x-1);
    }
}

constexpr int pow_integral(int x, size_t n)
{
    if(n==0) {
        return 1;
    }else{
        return x * pow_integral(x,n-1);
    }
}

constexpr size_t choose(size_t n, size_t x)
{
    if(n < x) return 0;
    return factorial(n) / (factorial(x) * factorial(n-x));
}

template<typename P>
constexpr P choose_by_factorial(size_t n, size_t x)
{
    return static_cast<P>(1.0) / (factorial(x) * factorial(n-x));
}

template<typename P, size_t K>
Eigen::Vector<P,K> powers(P time_u)
{
    Eigen::Vector<P,K> ts;
    ts[0] = static_cast<P>(1.0);

    if constexpr (K >= 2) {
        ts[1] = time_u;
    }

    for (size_t i=2; i < K; ++i) {
        ts[i] = ts[i-1] * time_u;
    }
    return ts;
}

template<typename P, size_t K>
Eigen::Matrix<P,K,Eigen::Dynamic> powers(Eigen::Matrix<P,1,Eigen::Dynamic> time_u)
{
    Eigen::Matrix<P,K,Eigen::Dynamic> ts(K,time_u.size());

    ts.row(0) = Eigen::Matrix<P,1,Eigen::Dynamic>::Constant(1, ts.cols(), 1.0);

    if constexpr (K >= 2) {
        ts.row(1) = time_u;
    }

    for (size_t i=2; i < K; ++i) {
        ts.row(i) = ts.row(i-1).array() * time_u.array();
    }
    return ts;
}

// Represent k_order spline basis as matrix coeffs.
// i.e:
//   coeffs = cardinal_basis_spline_matrix<P,K>() * time_powers<P,K>(time_u)
//   value_u = weights^T coeefs
template<typename P, size_t K>
Eigen::Matrix<P,K,K> cardinal_basis_spline_matrix()
{
    Eigen::Matrix<P,K,K> M = Eigen::Matrix<P,K,K>::Zero();

    constexpr size_t K_minus = K-1;

    for(int c=0; c < K; ++c) {
        const P fac = choose_by_factorial<P>(K_minus, K_minus-c);

        for(int r=0; r < K; ++r) {
            P& el = M(r,c);
            for(int s=r; s < K; ++s) {
                el += pow_integral(-1, s-r) * (int)choose(K, s-r) * pow_integral(K_minus-s, K_minus-c);
            }
            el *= fac;
        }
    }

    return M;
}

template<typename P, size_t K>
struct SplineConstants
{
    static const Eigen::Matrix<P,K,K>& cardinal_matrix()
    {
        const static Eigen::Matrix<P,K,K> M = cardinal_basis_spline_matrix<P,K>();
        return M;
    }
};

// Evaluate Kth order spline \p control_points at \p t
// Each control point is a column vector in \p control_points
template<typename P, size_t K, typename C = P>
Eigen::VectorX<C> eval_cardinal_basis_spline(
    const P control_point_interval,
    const Eigen::MatrixX<C>& control_points,
    const P t )
{
    const P tc = t / control_point_interval;
    const size_t i(tc);
    assert(i <= control_points.cols() - K);
    assert(K <= control_points.cols());

    const P u = tc - i;
    const Eigen::Vector<P,K> t_pows = powers<P,K>(u);
    const Eigen::Vector<P,K> coeffs = SplineConstants<double,K>::cardinal_matrix() * t_pows;

    return control_points.template middleCols<K>(i) * coeffs;
}

// Evaluate Kth order spline \p control_points at \p t
// Each control point is a column vector in \p control_points
template<typename P, size_t K, typename C = P>
Eigen::MatrixX<C> eval_cardinal_basis_spline(
    const P control_point_interval,
    const Eigen::MatrixX<C>& control_points,
    const Eigen::Matrix<P,1,Eigen::Dynamic>& ts )
{
    assert(K <= control_points.cols());

    const Eigen::Matrix<P,1,Eigen::Dynamic> tcs = ts / control_point_interval;
    const Eigen::Matrix<int,1,Eigen::Dynamic> is = tcs.template cast<int>();
    const Eigen::Matrix<P,1,Eigen::Dynamic> us = tcs - is.template cast<P>();
    const Eigen::Matrix<P,K,Eigen::Dynamic> t_pows = powers<P,K>(us);
    const Eigen::Matrix<P,K,Eigen::Dynamic> coeffs = SplineConstants<double,K>::cardinal_matrix() * t_pows;

    Eigen::MatrixX<C> R(control_points.rows(), ts.cols());
    for(size_t j=0; j < ts.cols(); ++j) {
        assert(is[j] + K <= control_points.cols());
        R.col(j) = control_points.template middleCols<K>(is[j]) * coeffs.col(j);
    }
    return R;
}

template<typename P, size_t K, typename C = P>
Eigen::MatrixX<C> fit_cardinal_basis_spline(
    const size_t num_control_points,
    const P control_point_interval,
    const Eigen::Matrix<P,1,Eigen::Dynamic>& ts,
    const Eigen::MatrixX<C>& ys
) {
    using namespace Eigen;

    assert(ts.cols() == ys.cols());
    const Matrix<P,1,Eigen::Dynamic> tcs = ts / control_point_interval;
    const Matrix<int,1,Eigen::Dynamic> is = tcs.template cast<int>();
    const Matrix<P,1,Dynamic> us = tcs - is.template cast<P>();
    const Matrix<P,K,Dynamic> t_pows = powers<P,K>(us);
    const Matrix<P,K,Dynamic> coeffs = SplineConstants<double,K>::cardinal_matrix() * t_pows;


    // Q is the sparse version of coeffs shifted against the global array of control points
    // Each sample is a function of K control points
    SparseMatrix<P> Q(num_control_points, ts.cols());
    Q.reserve(VectorXi::Constant(ts.cols(), K));

    // Fill in the sparse version
    for(size_t j=0; j < ts.cols(); ++j) {
        assert(is[j] + K <= num_control_points);
        for(size_t k=0; k < K; ++k) {
            Q.coeffRef(is[j]+k,j) = coeffs(k,j);
        }
//        // segment probably wont work...
//        Q.col(j).template segment<K>(is[j]) = coeffs.col(j);
    }
    Q.makeCompressed();

//    // Dense version
//    Eigen::MatrixX<P> Q(num_control_points, ts.cols());
//    Q.setZero();
//    for(size_t j=0; j < ts.cols(); ++j) {
//        Q.col(j).template segment<K>(is[j]) = coeffs.col(j);
//    }
//    Eigen::MatrixX<C> control_points = Q.transpose().bdcSvd(ComputeThinU | ComputeThinV).solve(ys.transpose()).transpose();

    // Sparse version. We have a sparse rectangular non-adjoint problem
    SparseQR<SparseMatrix<double,RowMajor>, COLAMDOrdering<int>> solver;
    solver.compute(Q.transpose());

    if(solver.info()!=Success) {
        throw std::runtime_error("Factorization failed.");
    }

    Eigen::MatrixX<C> control_points = solver.solve(ys.transpose()).transpose();

    if(solver.info()!=Success) {
        throw std::runtime_error("Solve failed.");
    }

    return control_points;
}

#include <pangolin/display/display.h>
#include <pangolin/plot/plotter.h>

void test()
{
    using namespace pangolin;
    using namespace Eigen;

    assert(factorial(0) == 1);
    assert(factorial(1) == 1);
    assert(factorial(7) == 5040);
    assert(factorial(10) == 3628800);
    assert(pow_integral(0,4) == 0);
    assert(pow_integral(7,0) == 1);
    assert(pow_integral(2,3) == 8);
    assert(pow_integral(-10,3) == -1000);
    assert(choose(5,2) == 10);
    assert(choose(0,6) == 0);
    assert(choose(11, 3) == 165);

    std::cout << SplineConstants<double,4>::cardinal_matrix() << std::endl;

    const size_t K = 4;
    const double control_point_interval = 2.0;
    const Eigen::Matrix<double,1,Eigen::Dynamic> ts({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}});
    const Eigen::Matrix<double,1,Eigen::Dynamic> ys({{1.5, 3.2, 3.1, 4.7, 8.4, 11.3, 14.0}});
    const int num_control_points = int(ts.maxCoeff() / control_point_interval) + K;

    const Eigen::MatrixXd cp = fit_cardinal_basis_spline<double,K,double>(num_control_points, control_point_interval, ts, ys);
    std::cout << cp << std::endl << std::endl;
    std::cout << ys << std::endl;
    std::cout << eval_cardinal_basis_spline<double,K,double>(control_point_interval, cp, ts) << std::endl;

    Eigen::Matrix<float,2,Eigen::Dynamic> raw_plot(2, ts.cols());
    raw_plot.row(0) = ts.cast<float>();
    raw_plot.row(1) = ys.cast<float>();

    const Eigen::Matrix<double,1,Eigen::Dynamic> ts2 = Eigen::VectorXd::LinSpaced(100, 0.0, 6.0).transpose();
    const Eigen::Matrix<double,1,Eigen::Dynamic> ys2 = eval_cardinal_basis_spline<double,K,double>(control_point_interval, cp, ts2);
    Eigen::Matrix<float,2,Eigen::Dynamic> spline_plot(2, ts2.cols());
    spline_plot.row(0) = ts2.cast<float>();
    spline_plot.row(1) = ys2.cast<float>();


    pangolin::CreateWindowAndBind("Main",640,480);
    pangolin::DataLog log_raw;
    pangolin::DataLog log_spline;
    pangolin::Plotter plot(&log_raw,0.0,10.0,0.0,20.0);

    log_raw.Log(raw_plot.rows(), raw_plot.data(), raw_plot.cols());
    log_spline.Log(spline_plot.rows(), spline_plot.data(), spline_plot.cols());
    plot.ClearSeries();
    plot.AddSeries("$0", "$1", DrawingModeLine, Colour::Unspecified(), "data", &log_raw);
    plot.AddSeries("$0", "$1", DrawingModeLine, Colour::Unspecified(), "spline", &log_spline);

    pangolin::DisplayBase().AddDisplay(plot);
    while( !pangolin::ShouldQuit() )
    {
        pangolin::FinishFrame();
    }
}
