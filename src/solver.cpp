#include <heat_solver_3d/solver.hpp>

#include <cmath>
#include <algorithm>

namespace heat_solver_3d {

mpi_state_t::mpi_state_t(const mpi_params_t &params, MPI_Comm comm)
{
    if (!params.is_valid()) {
        throw solver_error_t("invalid MPI parameters");
    }

    original_comm = comm;

    int size;
    CHECKED_MPI(MPI_Comm_size(comm, &size));

    dims[0] = params.grid_x_split;
    dims[1] = params.grid_y_split;
    CHECKED_MPI(MPI_Dims_create(size, 2, dims.data()));

    int periods[2] = {0, 0};
    CHECKED_MPI(MPI_Cart_create(comm, 2, dims.data(), periods, 0, &cart_comm));
    CHECKED_MPI(MPI_Cart_get(cart_comm, 2, dims.data(), periods, coords.data()));

    int remain_dims[2];
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    CHECKED_MPI(MPI_Cart_sub(cart_comm, remain_dims, &pipeline_comms[0]));
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    CHECKED_MPI(MPI_Cart_sub(cart_comm, remain_dims, &pipeline_comms[1]));

    CHECKED_MPI(MPI_Comm_dup(cart_comm, &border_exchange_comms[0]));
    CHECKED_MPI(MPI_Comm_dup(cart_comm, &border_exchange_comms[1]));
    CHECKED_MPI(MPI_Comm_dup(cart_comm, &save_comm));
}

mpi_state_t::~mpi_state_t()
{
    MPI_Comm_free(&save_comm);
    MPI_Comm_free(&border_exchange_comms[1]);
    MPI_Comm_free(&border_exchange_comms[0]);
    MPI_Comm_free(&pipeline_comms[1]);
    MPI_Comm_free(&pipeline_comms[0]);
    MPI_Comm_free(&cart_comm);
}

solver_t::solver_t(const equation_params_t &equation_params,
                   const grid_params_t &grid_params,
                   const mpi_params_t &mpi_params,
                   MPI_Comm comm,
                   const double epsilon)
    : _equation_params(equation_params)
    , _grid_params(grid_params)
    , _mpi_params(mpi_params)
    , _epsilon(epsilon)
    , _handler(nullptr)
    , _mpi_state(mpi_params, comm)
    , _current_substep(0)
{
    if (!_equation_params.is_valid()) {
        throw solver_error_t("invalid equation parameters");
    }
    if (!_grid_params.is_valid()) {
        throw solver_error_t("invalid grid parameters");
    }
    if (_epsilon <= 0) {
        throw solver_error_t("invalid epsilon");
    }
    _init_dims();
    _init_mpi_types();
    _alloc_layers();
    _alloc_buffers();
}

solver_t::~solver_t()
{
    _free_mpi_types();
}

handler_t * solver_t::handler() const
{
    return _handler;
}

void solver_t::set_handler(handler_t *handler)
{
    _handler = handler;
}

size_t solver_t::current_step() const
{
    return _current_substep / 3;
}

double solver_t::current_t() const
{
    return _get_t(_current_substep);
}

void solver_t::run()
{
    _set_starting_grid_values();
    _current_substep = 0;
    while (_current_substep < _dims[3] - 1) {
        if (_handler && !_handler->handle(*this)) {
            return;
        }

        // To 1/3 by x
        _step_1();
        ++_current_substep;

        // To 2/3 by y
        _step_2();
        ++_current_substep;

        // To 1   by z
        _step_3();
        ++_current_substep;

        _layers[0].swap(_layers[1]);
        _set_x_borders(_layers[0], _current_substep);
        _set_y_borders(_layers[0], _current_substep);
        _set_z_borders(_layers[0], _current_substep);
    }
}

void solver_t::save_local_grid_state(std::ostream &os) const
{
    // x, y, z, t local limits
    for (const auto &val_cache: _dim_val_caches) {
        os.write(reinterpret_cast<const char *>(&val_cache.front()),
                 sizeof(double));
        os.write(reinterpret_cast<const char *>(&val_cache.back()),
                 sizeof(double));
    }
    const double t = _get_t(_current_substep);
    os.write(reinterpret_cast<const char *>(&t), sizeof(t));
    for (const size_t dim : _dims) {
        uint64_t tmp_dim = static_cast<uint64_t>(dim);
        os.write(reinterpret_cast<const char *>(&tmp_dim),
                 sizeof(tmp_dim));
    }
    _layers[0].save(os, false);
}

void solver_t::save_full_grid_state(std::ostream &os, const int save_rank) const
{
    int proc_rank;
    CHECKED_MPI(MPI_Comm_rank(_mpi_state.save_comm, &proc_rank));

    if (proc_rank != save_rank) {
        CHECKED_MPI(MPI_Send(_layers[0].data(), _layers[0].size(),
                             MPI_DOUBLE, 0, 0, _mpi_state.save_comm));
        return;
    }

    if (proc_rank == save_rank) {
        os.write(reinterpret_cast<const char *>(_equation_params.x_limits.data()),
                 _equation_params.x_limits.size() * sizeof(double));
        os.write(reinterpret_cast<const char *>(_equation_params.y_limits.data()),
                 _equation_params.y_limits.size() * sizeof(double));
        os.write(reinterpret_cast<const char *>(_equation_params.z_limits.data()),
                 _equation_params.z_limits.size() * sizeof(double));
        os.write(reinterpret_cast<const char *>(_equation_params.t_limits.data()),
                 _equation_params.t_limits.size() * sizeof(double));
        const double t = _get_t(_current_substep);
        os.write(reinterpret_cast<const char *>(&t), sizeof(t));
        for (const size_t dim : _total_dims) {
            uint64_t tmp_dim = static_cast<uint64_t>(dim);
            os.write(reinterpret_cast<const char *>(&tmp_dim),
                     sizeof(tmp_dim));
        }
        size_t base_pos = os.tellp();

        layer_t tmp_layer;
        std::array<size_t, 3> dims;
        std::array<size_t, 2> offsets;
        std::array<int, 2> coords;
        for (coords[0] = 0; coords[0] < _mpi_state.dims[0]; ++coords[0]) {
            for (coords[1] = 0; coords[1] < _mpi_state.dims[1]; ++coords[1]) {
                _get_space_dims_and_offsets_for_coords(coords, dims, offsets);

                if (coords == _mpi_state.coords) {
                    _save_layer_from_full_grid(os, base_pos, coords, offsets, _layers[0]);
                    continue;
                }

                tmp_layer.adjust_dims(dims);
                int rank;
                CHECKED_MPI(MPI_Cart_rank(_mpi_state.save_comm,
                                          coords.data(), &rank));
                CHECKED_MPI(MPI_Recv(tmp_layer.data(), tmp_layer.size(),
                                     MPI_DOUBLE, rank, 0, _mpi_state.save_comm,
                                     MPI_STATUS_IGNORE));
                _save_layer_from_full_grid(os, base_pos, coords, offsets, tmp_layer);
            }
        }
    }
}

double solver_t::_get_x(const size_t i) const
{
    return _dim_val_caches[0][i];
}

double solver_t::_get_y(const size_t j) const
{
    return _dim_val_caches[1][j];
}

double solver_t::_get_z(const size_t k) const
{
    return _dim_val_caches[2][k];
}

double solver_t::_get_t(const size_t substep) const
{
    return _dim_val_caches[3][substep];
}

void solver_t::_init_dims()
{
    double dim_add_f;
    size_t dim_add_u;

    // Calculating the X dimensions
    dim_add_f = (_equation_params.x_limits[1] - _equation_params.x_limits[0]) /
            _grid_params.space_step;
    dim_add_u = static_cast<size_t>(std::round(dim_add_f));
    if (std::abs(dim_add_f - dim_add_u) > _epsilon) {
        throw solver_error_t("cannot evenly split the X dimension");
    }
    _total_dims[0] = 1 + dim_add_u;
    const size_t x_offset = _set_split_dim(_mpi_state.coords, 0, _dims[0]);
    const double x_0 = _equation_params.x_limits[0] +
            _grid_params.space_step * x_offset;

    // Calculating the Y dimensions
    dim_add_f = (_equation_params.y_limits[1] - _equation_params.y_limits[0]) /
            _grid_params.space_step;
    dim_add_u = static_cast<size_t>(std::round(dim_add_f));
    if (std::abs(dim_add_f - dim_add_u) > _epsilon) {
        throw solver_error_t("cannot evenly split the Y dimension");
    }
    _total_dims[1] = 1 + dim_add_u;
    const size_t y_offset = _set_split_dim(_mpi_state.coords, 1, _dims[1]);
    const double y_0 = _equation_params.y_limits[0] +
            _grid_params.space_step * y_offset;

    // Calculating the Z dimensions
    dim_add_f = (_equation_params.z_limits[1] - _equation_params.z_limits[0]) /
            _grid_params.space_step;
    dim_add_u = static_cast<size_t>(std::round(dim_add_f));
    if (std::abs(dim_add_f - dim_add_u) > _epsilon) {
        throw solver_error_t("cannot evenly split the Z dimension");
    }
    _total_dims[2] = _dims[2] = 1 + dim_add_u;
    double z_0 = _equation_params.z_limits[0];

    // Calculating the T dimensions
    // In substeps
    dim_add_f = (_equation_params.t_limits[1] - _equation_params.t_limits[0]) /
            _grid_params.time_step;
    dim_add_u = static_cast<size_t>(std::round(dim_add_f));
    if (std::abs(dim_add_f - dim_add_u) > _epsilon) {
        throw solver_error_t("cannot evenly split the T dimension");
    }
    // Tripling, since we have 3 substeps per step
    _total_dims[3] = _dims[3] = 1 + 3 * dim_add_u;
    double t_0 = _equation_params.t_limits[0];

    // Setting up all of the dimension caches
    for (size_t dim = 0; dim < _dims.size(); ++dim) {
        _dim_val_caches[dim].resize(_dims[dim]);
    }
    for (size_t i = 0; i < _dims[0]; ++i) {
        _dim_val_caches[0][i] = x_0 + i * _grid_params.space_step;
    }
    for (size_t j = 0; j < _dims[1]; ++j) {
        _dim_val_caches[1][j] = y_0 + j * _grid_params.space_step;
    }
    for (size_t k = 0; k < _dims[2]; ++k) {
        _dim_val_caches[2][k] = z_0 + k * _grid_params.space_step;
    }
    for (size_t substep = 0; substep < _dims[3]; ++substep) {
        _dim_val_caches[3][substep] = t_0 + substep * _grid_params.time_step / 3.0;
    }
}

void solver_t::_init_mpi_types()
{
    CHECKED_MPI(MPI_Type_contiguous(_dims[1] * _dims[2], MPI_DOUBLE, &_border_exchange_types[0]));
    CHECKED_MPI(MPI_Type_commit(&_border_exchange_types[0]));
    CHECKED_MPI(MPI_Type_vector(_dims[0], _dims[2], _dims[1] * _dims[2], MPI_DOUBLE, &_border_exchange_types[1]));
    CHECKED_MPI(MPI_Type_commit(&_border_exchange_types[1]));
}

void solver_t::_alloc_layers()
{
    for (layer_t &layer : _layers) {
        layer.adjust_dims(_dims.data());
    }
}

void solver_t::_alloc_buffers()
{
    // TODO: Check for excess memory
    size_t max_dimension = *std::max_element(_dims.begin(), _dims.begin() + 3);
    // Step 3
    size_t size = max_dimension * 3;
    // Step 1/2
    size = std::max(size, _mpi_params.pipeline_limit * std::max(_dims[0], _dims[1]) * 3);
    _tmp_buf.resize(size);
    // Need delta buf only for quasilinear equations;
    if (_equation_params.is_quasilinear) {
        _delta_buf.resize(std::max(_dims[0], _dims[1]) * _dims[2]);
    }
}

void solver_t::_free_mpi_types()
{
    MPI_Type_free(&_border_exchange_types[1]);
    MPI_Type_free(&_border_exchange_types[0]);
}

void solver_t::_set_starting_grid_values()
{
    layer_t &layer = _layers[0];
    const double t0 = _equation_params.t_limits[0];
    for (size_t i = 0; i < _dims[0]; ++i) {
        const double x = _get_x(i);
        for (size_t j = 0; j < _dims[1]; ++j) {
            const double y = _get_y(j);
            for (size_t k = 0; k < _dims[2]; ++k) {
                const double z = _get_z(k);
                layer(i, j, k) = _equation_params.u0(x, y, z, t0);
            }
        }
    }
}

void solver_t::_set_x_borders(layer_t &layer, const size_t substep)
{
    const double t = _get_t(substep);
    for (size_t j = 0; j < _dims[1]; ++j) {
        const double y = _get_y(j);
        for (size_t k = 0; k < _dims[2]; ++k) {
            const double z = _get_z(k);
            if (_mpi_state.coords[0] == 0) {
                layer(0, j, k) = _equation_params.gamma[0](y, z, t);
            }
            if (_mpi_state.coords[0] == _mpi_state.dims[0] - 1) {
                layer(_dims[0] - 1, j, k) = _equation_params.gamma[1](y, z, t);
            }
        }
    }
}

void solver_t::_set_y_borders(layer_t &layer, const size_t substep)
{
    const double t = _get_t(substep);
    for (size_t i = 0; i < _dims[0]; ++i) {
        const double x = _get_x(i);
        for (size_t k = 0; k < _dims[2]; ++k) {
            const double z = _get_z(k);
            if (_mpi_state.coords[1] == 0) {
                layer(i, 0, k) = _equation_params.gamma[2](x, z, t);
            }
            if (_mpi_state.coords[1] == _mpi_state.dims[1] - 1) {
                layer(i, _dims[1] - 1, k) = _equation_params.gamma[3](x, z, t);
            }
        }
    }
}

void solver_t::_set_z_borders(layer_t &layer, const size_t substep)
{
    const double t = _get_t(substep);
    for (size_t i = 0; i < _dims[0]; ++i) {
        const double x = _get_x(i);
        for (size_t j = 0; j < _dims[1]; ++j) {
            const double y = _get_y(j);
            layer(i, j, 0) = _equation_params.gamma[4](x, y, t);
            layer(i, j, _dims[2] - 1) = _equation_params.gamma[5](x, y, t);
        }
    }
}

void solver_t::_exchange_x_border(layer_t &layer)
{
    MPI_Datatype type = _border_exchange_types[0];
    MPI_Comm comm = _mpi_state.border_exchange_comms[0];

    int prev_proc, next_proc;
    CHECKED_MPI(MPI_Cart_shift(comm, 0, 1, &prev_proc, &next_proc));
    MPI_Request reqs[4];

    // Left
    CHECKED_MPI(MPI_Isend(&layer(1, 0, 0), 1, type,
                          prev_proc, 0, comm, reqs));
    CHECKED_MPI(MPI_Irecv(&layer(0, 0, 0), 1, type,
                          prev_proc, 0, comm, reqs + 1));

    // Right
    CHECKED_MPI(MPI_Isend(&layer(_dims[0] - 2, 0, 0), 1, type,
                          next_proc, 0, comm, reqs + 2));
    CHECKED_MPI(MPI_Irecv(&layer(_dims[0] - 1, 0, 0), 1, type,
                          next_proc, 0, comm, reqs + 3));

    CHECKED_MPI(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE));
}

void solver_t::_exchange_y_border(layer_t &layer)
{
    MPI_Datatype type = _border_exchange_types[1];
    MPI_Comm comm = _mpi_state.border_exchange_comms[1];

    int prev_proc, next_proc;
    CHECKED_MPI(MPI_Cart_shift(comm, 1, 1, &prev_proc, &next_proc));
    MPI_Request reqs[4];

    // Left
    CHECKED_MPI(MPI_Isend(&layer(0, 1, 0), 1, type,
                          prev_proc, 0, comm, reqs));
    CHECKED_MPI(MPI_Irecv(&layer(0, 0, 0), 1, type,
                          prev_proc, 0, comm, reqs + 1));

    // Right
    CHECKED_MPI(MPI_Isend(&layer(0, _dims[1] - 2, 0), 1, type,
                          next_proc, 0, comm, reqs + 2));
    CHECKED_MPI(MPI_Irecv(&layer(0, _dims[1] - 1, 0), 1, type,
                          next_proc, 0, comm, reqs + 3));

    CHECKED_MPI(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE));
}

double solver_t::_second_deriv_x(const layer_t &layer,
                                 const std::array<size_t, 3> &coor,
                                 const double t) const
{
    const double u_l = layer(coor[0] - 1, coor[1], coor[2]);
    const double u_c = layer(coor);
    const double u_r = layer(coor[0] + 1, coor[1], coor[2]);;

    const double y = _get_y(coor[1]);
    const double z = _get_z(coor[2]);
    const double g_l = _equation_params.g[0](_get_x(coor[0] - 1), y, z, t, u_l);
    const double g_c = _equation_params.g[0](_get_x(coor[0]), y, z, t, u_c);
    const double g_r = _equation_params.g[0](_get_x(coor[0] + 1), y, z, t, u_r);

    return ((g_r + g_c) * (u_r - u_c) - (g_c + g_l) * (u_c - u_l)) /
           (2.0 * _grid_params.space_step * _grid_params.space_step);
}

double solver_t::_second_deriv_y(const layer_t &layer,
                                 const std::array<size_t, 3> &coor,
                                 const double t) const
{
    const double u_l = layer(coor[0], coor[1] - 1, coor[2]);
    const double u_c = layer(coor);
    const double u_r = layer(coor[0], coor[1] + 1, coor[2]);;

    const double x = _get_x(coor[0]);
    const double z = _get_z(coor[2]);
    const double g_l = _equation_params.g[1](x, _get_y(coor[1] - 1), z, t, u_l);
    const double g_c = _equation_params.g[1](x, _get_y(coor[1]), z, t, u_c);
    const double g_r = _equation_params.g[1](x, _get_y(coor[1] + 1), z, t, u_r);

    return ((g_r + g_c) * (u_r - u_c) - (g_c + g_l) * (u_c - u_l)) /
           (2.0 * _grid_params.space_step * _grid_params.space_step);
}

double solver_t::_second_deriv_z(const layer_t &layer,
                                 const std::array<size_t, 3> &coor,
                                 const double t) const
{
    const double u_l = layer(coor[0], coor[1], coor[2] - 1);
    const double u_c = layer(coor);
    const double u_r = layer(coor[0], coor[1], coor[2] + 1);;

    const double x = _get_x(coor[0]);
    const double y = _get_y(coor[1]);
    const double g_l = _equation_params.g[2](x, y, _get_z(coor[2] - 1), t, u_l);
    const double g_c = _equation_params.g[2](x, y, _get_z(coor[2]), t, u_c);
    const double g_r = _equation_params.g[2](x, y, _get_z(coor[2] + 1), t, u_r);

    return ((g_r + g_c) * (u_r - u_c) - (g_c + g_l) * (u_c - u_l)) /
            (2.0 * _grid_params.space_step * _grid_params.space_step);
}

double solver_t::_gen_a_i(const double *g,
                          const std::array<size_t, 3> coor,
                          const size_t dim)
{
    return _grid_params.time_step * (g[coor[dim] - 1] + g[coor[dim]]);
}

double solver_t::_gen_b_i(const double *g,
                          const std::array<size_t, 3> coor,
                          const size_t dim)
{
    return -4 * _grid_params.space_step * _grid_params.space_step -
            _grid_params.time_step * (
                g[coor[dim] - 1] + 2 * g[coor[dim]] + g[coor[dim] + 1]
            );
}

double solver_t::_gen_c_i(const double *g,
                          const std::array<size_t, 3> coor,
                          const size_t dim)
{
    return _grid_params.time_step * (g[coor[dim] + 1] + g[coor[dim]]);
}

double solver_t::_gen_d_i(const std::array<size_t, 3> coor,
                          const size_t substep)
{
    const size_t substep_offset = substep % 3;
    const double t = _get_t(substep - substep_offset);
    switch (substep_offset) {
    case 0:
        return -2 * _grid_params.space_step * _grid_params.space_step * (
            _grid_params.time_step * (
                _second_deriv_x(_layers[0], coor, t) +
                2 * _second_deriv_y(_layers[0], coor, t) +
                2 * _second_deriv_z(_layers[0], coor, t) +
                2 * _equation_params.f(_get_x(coor[0]), _get_y(coor[1]),
                                       _get_z(coor[2]), t, _layers[0](coor))
            ) + 2 * _layers[0](coor)
        );
    case 1:
        return 2 * _grid_params.space_step * _grid_params.space_step * (
            _grid_params.time_step * _second_deriv_y(_layers[0], coor, t) -
                2 * _layers[1](coor)
        );
    case 2:
        return 2 * _grid_params.space_step * _grid_params.space_step * (
            _grid_params.time_step * _second_deriv_z(_layers[0], coor, t) -
                2 * _layers[2](coor)
        );
    default:
        // Should not happen
        return 0.0;
    }
}

double solver_t::_process_row(layer_t &next_layer,
                              std::array<size_t, 3> coor,
                              const size_t dim,
                              const size_t substep)
{
    // Getting t for the next layer
    const double t = _get_t(substep + 1);

    // Init conductivities for future
    double *g = _tmp_buf.data();
    coor[dim] = 0;
    while (coor[dim] < _dims[dim]) {
        g[coor[dim]] = _equation_params.g[dim](
            _get_x(coor[0]), _get_y(coor[1]), _get_z(coor[2]), t, next_layer(coor)
        );
        ++coor[dim];
    }

    double *c = g + _tmp_buf.size() / 3;
    coor[dim] = 1;
    double c_i = _gen_c_i(g, coor, dim);
    double b_i = _gen_b_i(g, coor, dim);
    double a_i;
    c[1] = c_i / b_i;
    coor[dim] = 2;
    while (coor[dim] < _dims[dim] - 1) {
        c_i = _gen_c_i(g, coor, dim);
        b_i = _gen_b_i(g, coor, dim);
        a_i = _gen_a_i(g, coor, dim);
        c[coor[dim]] = c_i / (b_i - a_i * c[coor[dim] - 1]);
        ++coor[dim];
    }

    double *d = c + _tmp_buf.size() / 3;

    coor[dim] = 1;
    double d_i = _gen_d_i(coor, substep);
    a_i = _gen_a_i(g, coor, dim);
    coor[dim] = 0;
    d_i -= a_i * next_layer(coor);

    coor[dim] = 1;
    b_i = _gen_b_i(g, coor, dim);
    d[1] = d_i / b_i;

    coor[dim] = _dims[dim] - 2;
    double d_n = _gen_d_i(coor, substep);
    c_i = _gen_c_i(g, coor, dim);
    coor[dim] = _dims[dim] - 1;
    d_n -= c_i * next_layer(coor);

    coor[dim] = 2;
    while (coor[dim] < _dims[dim] - 2) {
        a_i = _gen_a_i(g, coor, dim);
        d_i = _gen_d_i(coor, substep);
        b_i = _gen_b_i(g, coor, dim);
        d[coor[dim]] = (d_i - a_i * d[coor[dim] - 1]) /
                       (b_i - a_i * c[coor[dim] - 1]);
        ++coor[dim];
    }

    a_i = _gen_a_i(g, coor, dim);
    b_i = _gen_b_i(g, coor, dim);

    /*
     * Calculating deltas only makes sense for quasilinear equations.
     * Linear equations will have max_delta = 0.
     */

    double prev_val = 0.0;
    if (_equation_params.is_quasilinear) {
        prev_val = next_layer(coor);
    }

    double new_val = (d_n - a_i * d[coor[dim] - 1]) /
                     (b_i - a_i * c[coor[dim] - 1]);
    next_layer(coor) = new_val;
    double max_delta = _equation_params.is_quasilinear
            ? std::abs(new_val - prev_val) : 0.0;
    while (coor[dim] > 1) {
        const double val_r = next_layer(coor);
        --coor[dim];

        new_val = d[coor[dim]] - c[coor[dim]] * val_r;

        if (_equation_params.is_quasilinear) {
            prev_val = next_layer(coor);
            const double delta = std::abs(new_val - prev_val);
            if (delta > max_delta) {
                max_delta = delta;
            }
        }

        next_layer(coor) = new_val;
    }

    return max_delta;
}

bool solver_t::_process_pipeline_step(layer_t &next_layer,
                                      const size_t outer_dim,
                                      const size_t inner_dim,
                                      const size_t substep)
{
    double *g = _tmp_buf.data();
    // Tuples of (b, c, d)
    double *eq_buf = _tmp_buf.data() + _dims[inner_dim];

    MPI_Comm comm = _mpi_state.pipeline_comms[substep % 3];
    int prev_proc, next_proc;
    CHECKED_MPI(MPI_Cart_shift(comm, 0, 1, &prev_proc, &next_proc));

    std::vector<MPI_Request> requests_down(_mpi_params.pipeline_limit, MPI_REQUEST_NULL);
    std::vector<MPI_Request> requests_up(_mpi_params.pipeline_limit, MPI_REQUEST_NULL);

    const double next_t = _get_t(substep + 1);
    std::array<size_t, 3> coor;
    coor[outer_dim] = 1;
    coor[2] = 1;
    const size_t calc_downto_inner = (_mpi_state.coords[inner_dim] == 0 ? 1 : 0);

    size_t global_iter = 0;
    const size_t max_global_iter = (_dims[outer_dim] - 2) * (_dims[2] - 2);
    while (global_iter < max_global_iter) {
        const size_t initial_global_iter = global_iter;
        const size_t initial_outer_coor = coor[outer_dim];
        const size_t initial_z_coor = coor[2];
        size_t i = 0;

        size_t iter = 0;
        const size_t max_iter = _mpi_params.pipeline_limit;

        // Downward zeroing part
        while (global_iter < max_global_iter && iter < max_iter) {
            if (!_equation_params.is_quasilinear ||
                    _delta_buf[global_iter] > _epsilon) {
                coor[inner_dim] = 0;
                while (coor[inner_dim] < _dims[inner_dim]) {
                    g[coor[inner_dim]] = _equation_params.g[inner_dim](
                            _get_x(coor[0]), _get_y(coor[1]), _get_z(coor[2]),
                            next_t, next_layer(coor));
                    ++coor[inner_dim];
                }

                coor[inner_dim] = 1;
                eq_buf[i + 3] = _gen_b_i(g, coor, inner_dim);
                eq_buf[i + 4] = _gen_c_i(g, coor, inner_dim);
                eq_buf[i + 5] = _gen_d_i(coor, substep);
                double a = _gen_a_i(g, coor, inner_dim);
                double rel;
                if (_mpi_state.coords[inner_dim] == 0) {
                    coor[inner_dim] = 0;
                    eq_buf[i + 5] -= a * next_layer(coor);
                } else {
                    CHECKED_MPI(MPI_Recv(eq_buf + i, 3, MPI_DOUBLE,
                                         prev_proc, iter, comm,
                                         MPI_STATUS_IGNORE));
                    rel = a / eq_buf[i];
                    eq_buf[i + 3] -= eq_buf[i + 1] * rel;
                    eq_buf[i + 5] -= eq_buf[i + 2] * rel;
                }
                i += 3;

                coor[inner_dim] = 2;
                while (coor[inner_dim] < _dims[inner_dim] - 1) {
                    i += 3;
                    a = _gen_a_i(g, coor, inner_dim);
                    rel = a / eq_buf[i - 3];
                    eq_buf[i] = _gen_b_i(g, coor, inner_dim) -
                            eq_buf[i - 2] * rel;
                    eq_buf[i + 1] = _gen_c_i(g, coor, inner_dim);
                    eq_buf[i + 2] = _gen_d_i(coor, substep) -
                            eq_buf[i - 1] * rel;
                    ++coor[inner_dim];
                }

                CHECKED_MPI(MPI_Isend(eq_buf + i, 3, MPI_DOUBLE,
                                      next_proc, iter, comm,
                                      requests_down.data() + iter));
                i += 3;
                ++iter;
            }

            ++global_iter;
            if (coor[2] != _dims[2] - 2) {
                ++coor[2];
            } else {
                coor[2] = 1;
                ++coor[outer_dim];
            }
        }

        // Reset state
        global_iter = initial_global_iter;
        iter = 0;
        coor[outer_dim] = initial_outer_coor;
        coor[2] = initial_z_coor;
        i = 3 * (_dims[inner_dim] - 1);

        // Upward calculating part
        while (global_iter < max_global_iter && iter < max_iter) {
            // _delta_buf is only used for quasilinear
            if (!_equation_params.is_quasilinear ||
                    _delta_buf[global_iter] > _epsilon) {
                double next_u;
                coor[inner_dim] = _dims[inner_dim] - 1;
                if (_mpi_state.coords[inner_dim] == _mpi_state.dims[inner_dim] - 1) {
                    next_u = next_layer(coor);
                    if (_equation_params.is_quasilinear) {
                        // If we are processing, then reset the delta,
                        // since otherwise it will stay DOUBLE_MAX
                        _delta_buf[global_iter] = 0.0;
                    }
                } else {
                    CHECKED_MPI(MPI_Recv(&next_u, 1, MPI_DOUBLE,
                                         next_proc, iter, comm,
                                         MPI_STATUS_IGNORE));

                    if (_equation_params.is_quasilinear) {
                        const double prev_val = next_layer(coor);
                        _delta_buf[global_iter] = std::abs(next_u - prev_val);
                    }

                    next_layer(coor) = next_u;
                }

                while (coor[inner_dim] > calc_downto_inner) {
                    --coor[inner_dim];
                    i -= 3;

                    next_u = (eq_buf[i + 2] - eq_buf[i + 1] * next_u) / eq_buf[i];

                    if (_equation_params.is_quasilinear) {
                        const double prev_val = next_layer(coor);
                        const double delta = std::abs(next_u - prev_val);
                        if (delta > _delta_buf[global_iter]) {
                            _delta_buf[global_iter] = delta;
                        }
                    }

                    next_layer(coor) = next_u;
                }

                coor[inner_dim] = 1;
                CHECKED_MPI(MPI_Isend(&next_layer(coor), 1, MPI_DOUBLE,
                                      prev_proc, iter, comm,
                                      requests_up.data() + iter));
                i += 3 * (_dims[inner_dim] - 1) + 3 * (_dims[inner_dim] - 1 - calc_downto_inner);
                ++iter;
            }

            ++global_iter;
            if (coor[2] != _dims[2] - 2) {
                ++coor[2];
            } else {
                coor[2] = 1;
                ++coor[outer_dim];
            }
        }

        // Wating for all comunications before starting
        // a new "pipeline" iteration
        CHECKED_MPI(MPI_Waitall(requests_down.size(), requests_down.data(),
                                MPI_STATUSES_IGNORE));
        CHECKED_MPI(MPI_Waitall(requests_up.size(), requests_up.data(),
                                MPI_STATUSES_IGNORE));
    }

    // _delta_buf is only used for quasilinear
    if (_equation_params.is_quasilinear) {
        CHECKED_MPI(MPI_Allreduce(MPI_IN_PLACE, _delta_buf.data(),
                                  max_global_iter, MPI_DOUBLE,
                                  MPI_MAX, comm));
        return std::any_of(_delta_buf.begin(),
                           _delta_buf.begin() + max_global_iter,
                           [this](double x){return x > this->_epsilon;});
    } else {
        return false;
    }
}

void solver_t::_step_1()
{
    // If quasilinear, we need to copy the previous layer as the first
    // approximation and init the delta buffer
    if (_equation_params.is_quasilinear) {
        _layers[1] = _layers[0];
        // Settings the delta to max, so it processes
        // the whole grid on the first iteration
        std::fill(_delta_buf.begin(), _delta_buf.end(),
                  std::numeric_limits<double>::max());
    }
    _set_x_borders(_layers[1], _current_substep + 1);

    bool not_finished;
    do {
        not_finished = _process_pipeline_step(_layers[1], 1, 0, _current_substep);
        _exchange_y_border(_layers[1]);
        // No need for state sync in a linear case, since not_finished will
        // always be false
        if (_equation_params.is_quasilinear) {
            MPI_Allreduce(MPI_IN_PLACE, &not_finished, 1,
                          MPI_CXX_BOOL, MPI_LOR, _mpi_state.cart_comm);
        }
    } while (not_finished);
}

void solver_t::_step_2()
{
    // If quasilinear, we need to copy the previous layer as the first
    // approximation and init the delta buffer
    if (_equation_params.is_quasilinear) {
        _layers[2] = _layers[1];
        // Settings the delta to max, so it processes
        // the whole grid on the first iteration
        std::fill(_delta_buf.begin(), _delta_buf.end(),
                  std::numeric_limits<double>::max());
    }
    _set_y_borders(_layers[2], _current_substep + 1);

    bool not_finished;
    do {
        not_finished = _process_pipeline_step(_layers[2], 0, 1, _current_substep);
        _exchange_x_border(_layers[2]);
        // No need for state sync in a linear case, since not_finished will
        // always be false
        if (_equation_params.is_quasilinear) {
            MPI_Allreduce(MPI_IN_PLACE, &not_finished, 1,
                          MPI_CXX_BOOL, MPI_LOR, _mpi_state.cart_comm);
        }
    } while (not_finished);
}

void solver_t::_step_3()
{
    if (_equation_params.is_quasilinear) {
        _layers[1] = _layers[2];
    }
    _set_z_borders(_layers[1], _current_substep + 1);
    std::array<size_t, 3> coor;
    const size_t min_coor_0 = (_mpi_state.coords[0] == 0 ? 1 : 0);
    const size_t max_coor_0 = (_mpi_state.coords[0] == _mpi_state.dims[0] - 1
            ? _dims[0] - 1 : _dims[0]);
    const size_t min_coor_1 = (_mpi_state.coords[1] == 0 ? 1 : 0);
    const size_t max_coor_1 = (_mpi_state.coords[1] == _mpi_state.dims[1] - 1
            ? _dims[1] - 1 : _dims[1]);
    for (coor[0] = min_coor_0; coor[0] < max_coor_0; ++coor[0]) {
        for (coor[1] = min_coor_1; coor[1] < max_coor_1; ++coor[1]) {
            double delta;
            do {
                delta = _process_row(_layers[1], coor, 2, _current_substep);
            } while (delta > _epsilon);
        }
    }
}

size_t solver_t::_set_split_dim(const std::array<int, 2> &coords,
                                                const size_t dim_num, size_t &out_dim) const
{
    const size_t calc_dims = _total_dims[dim_num] - 2;
    const size_t mandatory_dim = calc_dims / _mpi_state.dims[dim_num];
    const int leftover_dim = calc_dims % _mpi_state.dims[dim_num];
    out_dim = mandatory_dim + (leftover_dim > coords[dim_num] ? 3 : 2);
    return mandatory_dim * coords[dim_num] + std::min(leftover_dim, coords[dim_num]);
}

void solver_t::_get_space_dims_and_offsets_for_coords(
        const std::array<int, 2> &coords,
        std::array<size_t, 3> &dims,
        std::array<size_t, 2> &offsets) const
{
    offsets[0] = _set_split_dim(coords, 0, dims[0]);
    offsets[1] = _set_split_dim(coords, 1, dims[1]);
    dims[2] = _total_dims[2];
}

void solver_t::_save_layer_from_full_grid(std::ostream &os,
                                          const size_t base_pos,
                                          const std::array<int, 2> &coords,
                                          const std::array<size_t, 2> &offsets,
                                          const layer_t &layer) const
{
    const size_t min_i = (coords[0] == 0) ? 0 : 1;
    const size_t max_i = (coords[0] == _mpi_state.dims[0] - 1)
            ? layer.x_dims() : layer.x_dims() - 1;
    const size_t min_j = (coords[1] == 0) ? 0 : 1;
    const size_t max_j = (coords[1] == _mpi_state.dims[1] - 1)
            ? layer.y_dims() : layer.y_dims() - 1;
    for (size_t i = min_i; i < max_i; ++i) {
        for (size_t j = min_j; j < max_j; ++j) {
            const size_t real_i = offsets[0] + i;
            const size_t real_j = offsets[1] + j;
            const size_t row_offset = _total_dims[2] * (_total_dims[1] * real_i + real_j);
            os.seekp(base_pos + row_offset * sizeof(double));
            const double *row = layer.get_z_row_ptr(i, j);
            os.write(reinterpret_cast<const char *>(row),
                     layer.z_dims() * sizeof(double));
        }
    }
}

}   // namespace heat_solver_3d
