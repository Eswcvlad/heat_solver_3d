#ifndef GRID_PARAMS_HPP
#define GRID_PARAMS_HPP

namespace heat_solver_3d
{

struct grid_params_t
{
    double space_step;
    double time_step;

    grid_params_t(const double space_step_, const double time_step_)
        : space_step(space_step_), time_step(time_step_)
    {
    }

    bool is_valid() const
    {
        return space_step > 0 && time_step > 0;
    }
};  // struct grid_params_t

}   // namespace heat_solver_3d

#endif  // GRID_PARAMS_HPP
