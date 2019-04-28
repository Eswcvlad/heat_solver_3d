#ifndef MPI_PARAMS_HPP
#define MPI_PARAMS_HPP

#include <cstddef>

namespace heat_solver_3d
{

struct mpi_params_t
{
    int grid_x_split = 0;
    int grid_y_split = 0;
    size_t pipeline_limit = size_t{100};

    bool is_valid() const
    {
        return grid_x_split >= 0 && grid_y_split >= 0 && pipeline_limit > 0;
    }
};  // struct mpi_params_t

}   // namespace heat_solver_3d

#endif  // MPI_PARAMS_HPP
