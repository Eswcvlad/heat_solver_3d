#include <heat_solver_3d/solver.hpp>

#include <mpi.h>

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>

// Creating the handler, that will log to stderr the current step and time
class log_handler_t : public heat_solver_3d::handler_t
{
public:
    log_handler_t(const int proc_rank) : _proc_rank(proc_rank)
    {

    }

    bool handle(heat_solver_3d::solver_t &solver) override
    {
        if (!_proc_rank) {
            std::cerr << "Step #" << solver.current_step()
                      << "; t = " << solver.current_t()
                      << ";" << std::endl;
        }
        return true;
    }

private:
    const int _proc_rank;
};

void run_solver()
{
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    // Setting up our handler
    log_handler_t handler(proc_rank);

    // Specifying the mpi related parameters:
    heat_solver_3d::mpi_params_t mpi_params;
    // How much we split the grid along the X axis
    // 0 - auto
    mpi_params.grid_x_split = 0;
    // How much we split the grid along the Y axis
    // 1 - we don't split along Y
    mpi_params.grid_y_split = 1;
    // During pipeline computaions for solving equations, how many equations
    // can buffer before we start computing the "upward" part
    // If in doubt - leave as default
    mpi_params.pipeline_limit = 100;

    // Specifying the equation we are going to solve
    heat_solver_3d::equation_params_t equation_params;
    // It is quasilinear
    equation_params.is_quasilinear = true;
    // Specifying the space and time limits
    // We will have a [0, +1] cube with [0, +60] time.
    equation_params.x_limits = {{0, 1}};
    equation_params.y_limits = {{0, 1}};
    equation_params.z_limits = {{0, 1}};
    equation_params.t_limits = {{0, 60}};
    // Setting diffusitivy very low at the border (i.e. aluminum cube in air)
    auto diffusivity = [](double x, double y, double z, double t, double u) {
        return (x > 0 && x < 1 && y > 0 && y < 1 && z > 0 && z < 1) ? 1.0 : 0.005;
    };
    equation_params.g[0] = diffusivity;
    equation_params.g[1] = diffusivity;
    equation_params.g[2] = diffusivity;
    // Setting external heat as "heat from the center",
    // that will increase with time, but is capped by current temp^3
    equation_params.f = [](double x, double y, double z, double t, double u) {
        if (x <= 0.45 && x >= 0.55 && y >= 0.45 && y <= 0.55 && z >= 0.45 && z <= 0.55) {
            return 0.0;
        }
        return std::min(500.0 * (1 + t / 60.0), u * u * u);
    };

    // Specifying the grid parameters
    // We will have a 101x101x101 grid with 1000 iterations along time
    heat_solver_3d::grid_params_t grid_params(1.0 / 100.0, 60.0 / 1000.0);

    // Statring the calculation process
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();
    heat_solver_3d::solver_t solver(equation_params,
                                    grid_params,
                                    mpi_params,
                                    MPI_COMM_WORLD);
    solver.set_handler(&handler);
    solver.run();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();

    if (!proc_rank) {
        typedef std::chrono::duration<double> double_seconds;
        std::cout << "Duration: "
                  << std::chrono::duration_cast<double_seconds>(end - start)
                     .count()
                  << " s\n";
    }

    // Saving result to the disk
    {
        std::ofstream os;
        if (!proc_rank) {
            os.open("results.bin", std::ios_base::out   |
                                   std::ios_base::trunc |
                                   std::ios_base::binary);
        }
        solver.save_full_grid_state(os, 0);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    run_solver();
    MPI_Finalize();
    return 0;
}
