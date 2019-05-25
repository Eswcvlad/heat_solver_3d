#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <heat_solver_3d/layer.hpp>
#include <heat_solver_3d/handler.hpp>
#include <heat_solver_3d/grid_params.hpp>
#include <heat_solver_3d/mpi_params.hpp>
#include <heat_solver_3d/equation_params.hpp>

#include <string>
#include <memory>
#include <ostream>
#include <stdexcept>

#include <mpi.h>

#if __GNUC__
    #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif

namespace heat_solver_3d
{

struct solver_error_t : public std::runtime_error
{
    explicit solver_error_t(const std::string &what_arg)
        : std::runtime_error(what_arg)
    {
    }

    explicit solver_error_t(const char *what_arg)
        : runtime_error(what_arg)
    {
    }
};

#define CHECKED_MPI(x) \
do { \
    if (UNLIKELY(x)) { \
        throw solver_error_t("mpi call error"); \
    } \
} while (0)

struct mpi_state_t
{
    std::array<int, 2> coords;
    std::array<int, 2> dims;

    MPI_Comm original_comm;
    MPI_Comm cart_comm;
    std::array<MPI_Comm, 2> pipeline_comms;
    std::array<MPI_Comm, 2> border_exchange_comms;
    MPI_Comm save_comm;

    mpi_state_t(const mpi_params_t &params, MPI_Comm comm);
    mpi_state_t(const mpi_state_t &) = delete;
    mpi_state_t & operator=(const mpi_state_t &) = delete;

    ~mpi_state_t();
};  // struct mpi_state_t

class solver_t
{
public:
    solver_t(const equation_params_t &equation_params,
             const grid_params_t &grid_params,
             const mpi_params_t &mpi_params,
             MPI_Comm comm,
             const double epsilon = 1e-6);
    solver_t(const solver_t &) = delete;
    solver_t & operator=(const solver_t &) = delete;

    ~solver_t();

    handler_t * handler() const;
    void set_handler(handler_t *handler);

    size_t current_step() const;
    double current_t() const;

    void run();

    void save_local_grid_state(std::ostream &os) const;
    void save_full_grid_state(std::ostream &os, const int save_rank = 0) const;

private:
    double _get_x(const size_t i) const;
    double _get_y(const size_t j) const;
    double _get_z(const size_t k) const;
    double _get_t(const size_t substep) const;

    // Initializes _dims
    // Required _mpi_state to be already initialized
    void _init_dims();

    // Initializes data types for border exchange
    // Requires _dims to be already initialized
    void _init_mpi_types();

    // Allocates memory for the layers
    // Requires _dims to be already initialized
    void _alloc_layers();

    // Allocates calculation buffers
    // Requires _dims to be already initialized
    void _alloc_buffers();

    // Frees the data types for border exchange
    void _free_mpi_types();

    // Initializes the 0-layer with the values from u0
    void _set_starting_grid_values();

    // Sets borders for the layer using the border functions
    void _set_x_borders(layer_t &result_layer, const size_t substep);
    void _set_y_borders(layer_t &result_layer, const size_t substep);
    void _set_z_borders(layer_t &result_layer, const size_t substep);

    // Exchanges borders between neighbouring processes
    void _exchange_x_border(layer_t &result_layer);
    void _exchange_y_border(layer_t &result_layer);

    // Calculating second derivative approximations
    double _second_deriv_x(const layer_t &result_layer,
                           const std::array<size_t, 3> &coor,
                           const double t) const;
    double _second_deriv_y(const layer_t &result_layer,
                           const std::array<size_t, 3> &coor,
                           const double t) const;
    double _second_deriv_z(const layer_t &result_layer,
                           const std::array<size_t, 3> &coor,
                           const double t) const;

    // Coefficients for Thomas algorithm
    double _gen_a_i(const double *g, const std::array<size_t, 3> coor,
                    const size_t dim);
    double _gen_b_i(const double *g, const std::array<size_t, 3> coor,
                    const size_t dim);
    double _gen_c_i(const double *g, const std::array<size_t, 3> coor,
                    const size_t dim);
    double _gen_d_i(const std::array<size_t, 3> coor, const size_t substep);

    // Thomas algorithm withtout pipelining and exchanges
    double _process_row(layer_t &next_layer,
                        std::array<size_t,3> coor,
                        const size_t dim,
                        const size_t substep);

    // Pipeline step (Thomas algorithm)
    // Returns true, if the precision is no yet enough (quasilinear)
    bool _process_pipeline_step(layer_t &next_layer,
                                const size_t outer_dim,
                                const size_t inner_dim,
                                const size_t substep);

    // Calculation substeps
    void _step_1();
    void _step_2();
    void _step_3();

    size_t _set_split_dim(const std::array<int, 2> &coords,
                          const size_t dim_num,
                          size_t &out_dim) const;
    void _get_space_dims_and_offsets_for_coords(
            const std::array<int, 2> &coords,
            std::array<size_t, 3> &dims,
            std::array<size_t, 2> &offsets) const;
    void _save_layer_from_full_grid(std::ostream &os,
                                    const size_t base_pos,
                                    const std::array<int, 2> &coords,
                                    const std::array<size_t, 2> &offsets,
                                    const layer_t &layer) const;

private:
    /*
     * Parameters block
     */

    const equation_params_t _equation_params;
    const grid_params_t _grid_params;
    const mpi_params_t _mpi_params;
    const double _epsilon;
    MPI_Comm _comm;
    handler_t *_handler;

    /*
     * State block
     */

    // Communicators and node details required
    mpi_state_t _mpi_state;
    // Additional types for border exchange
    std::array<MPI_Datatype, 2> _border_exchange_types;
    // x, y, z, t dimensions for current node
    std::array<size_t, 4> _dims;
    // x, y, z, t dimensions for the whole grid
    std::array<size_t, 4> _total_dims;
    // Computational layers: next/current step layer + 2 temporary ones
    std::array<layer_t, 3> _layers;
    size_t _current_substep;

    /*
     * Cache and buffers block
     */

    // Precomputed 2.0 * _grid_params.space_step * _grid_params.space_step
    double _space_step_sq_2x;

    // Caches for get_x, get_y, get_z, get_t
    std::array<std::vector<double>, 4> _dim_val_caches;
    // Buffer for Thomas algorithm vars
    std::vector<double> _tmp_buf;
    // Buffer for deltas on step 1 and 2, when the equation is quasilinear
    std::vector<double> _delta_buf;
};  // class solver_t

}   // namespace heat_solver_3d

#endif  // SOLVER_HPP
