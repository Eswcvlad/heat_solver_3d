#ifndef EQUATION_PARAMS_HPP
#define EQUATION_PARAMS_HPP

#include <array>
#include <functional>

namespace heat_solver_3d
{

typedef std::function<double(double, double, double)> xyt_func_t;
typedef std::function<double(double, double, double, double)> xyzt_func_t;
typedef std::function<double(double, double, double, double, double)> xyztu_func_t;

struct equation_params_t
{
    bool is_quasilinear = true;

    std::array<double, 2> x_limits = {{0, 1}};
    std::array<double, 2> y_limits = {{0, 1}};
    std::array<double, 2> z_limits = {{0, 1}};
    std::array<double, 2> t_limits = {{0, 1}};

    std::array<xyztu_func_t, 3> g = {{
        [](double x, double y, double z, double t, double u)
        { (void)x; (void)y; (void)z; (void)t; (void)u; return 1.0; },
        [](double x, double y, double z, double t, double u)
        { (void)x; (void)y; (void)z; (void)t; (void)u; return 1.0; },
        [](double x, double y, double z, double t, double u)
        { (void)x; (void)y; (void)z; (void)t; (void)u; return 1.0; }
    }};

    xyztu_func_t f = [](double x, double y, double z, double t, double u)
    { (void)x; (void)y; (void)z; (void)t; (void)u; return 0.0; };

    // Lowest level init
    xyzt_func_t u0 = [](double x, double y, double z, double t)
    { (void)x; (void)y; (void)z; (void)t; return 20.0; };

    // Refer to task
    std::array<xyt_func_t, 6> gamma = {{
        [](double y, double z, double t)
        { (void)y; (void)z; (void)t; return 20.0; },
        [](double y, double z, double t)
        { (void)y; (void)z; (void)t; return 20.0; },
        [](double x, double z, double t)
        { (void)x; (void)z; (void)t; return 20.0; },
        [](double x, double z, double t)
        { (void)x; (void)z; (void)t; return 20.0; },
        [](double x, double y, double t)
        { (void)x; (void)y; (void)t; return 20.0; },
        [](double x, double y, double t)
        { (void)x; (void)y; (void)t; return 20.0; }
    }};

    bool is_valid() const
    {
        return x_limits[0] < x_limits[1] &&
               y_limits[0] < y_limits[1] &&
               z_limits[0] < z_limits[1] &&
               t_limits[0] < t_limits[1];
    }
};  // struct equation_params_t

}   // namespace heat_solver_3d

#endif  // EQUATION_PARAMS_HPP
