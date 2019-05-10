#include <heat_solver_3d/solver.hpp>

#include <mpi.h>

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>

// Creating the handler, that will log to stderr the current step and time
class laser_handler_t : public heat_solver_3d::handler_t
{
public:
    laser_handler_t(const int proc_rank, const std::string file_path_prefix)
        : _proc_rank(proc_rank)
        , _file_path_prefix(file_path_prefix)
    {

    }

    bool handle(heat_solver_3d::solver_t &solver) override
    {
        log(solver);
        write_state(solver);
        return true;
    }

private:
    void log(const heat_solver_3d::solver_t &solver) const
    {
        if (!_proc_rank) {
            std::cerr << "Step #" << solver.current_step()
                      << "; t = " << solver.current_t()
                      << ";" << std::endl;
        }
    }

    void write_state(const heat_solver_3d::solver_t &solver) const
    {
        if (solver.current_step() % 20) {
            return;
        }

        std::ofstream os;
        if (!_proc_rank) {
            os.open(_file_path_prefix + '.' + std::to_string(solver.current_step()),
                    std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        }
        solver.save_full_grid_state(os);
    }

    const int _proc_rank;
    const std::string _file_path_prefix;
};

void run_solver()
{
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    // Setting up our handler
    laser_handler_t handler(proc_rank, "laser-res.bin");

    // Specifying the mpi related parameters:
    heat_solver_3d::mpi_params_t mpi_params;
    // How much we split the grid along the X axis
    // 0 - auto
    mpi_params.grid_x_split = 0;
    // How much we split the grid along the Y axis
    // 0 - auto
    mpi_params.grid_y_split = 0;
    // During pipeline computaions for solving equations, how many equations
    // can buffer before we start computing the "upward" part
    // If in doubt - leave as default
    mpi_params.pipeline_limit = 100;

    // Specifying the equation we are going to solve
    heat_solver_3d::equation_params_t equation_params;
    // It is NOT quasilinear
    equation_params.is_quasilinear = false;
    // Specifying the space and time limits
    // Space will be [-0.01, 1.01]
    // The cube will be [0, 1] in space
    // Only the center part will be metal
    // Time: [0, 60]
    equation_params.x_limits = {{-0.01, 1.01}};
    equation_params.y_limits = {{-0.01, 1.01}};
    equation_params.z_limits = {{-0.01, 1.01}};
    equation_params.t_limits = {{0, 60}};
    // Space border is air
    // Cube border is a polymer
    // Cube insides are iron
    auto diffusivity = [](double x, double y, double z, double t, double u) {
        const double air_diffusivity = 0.000019;
        const double pvc_diffusivity = 0.00000008;
        const double iron_diffusivity = 0.000023;
        if (x < 0 || x > 1 || y < 0 || y > 1 || z < 0 || z > 1) {
            return air_diffusivity;
        }
        if (x < 0.05 || x > 0.95 || y < 0.05 || y > 0.95 || z < 0.05 || z > 0.95) {
            return pvc_diffusivity;
        }
        return iron_diffusivity;
    };
    equation_params.g[0] = diffusivity;
    equation_params.g[1] = diffusivity;
    equation_params.g[2] = diffusivity;
    // Running the laser along the whole border of the cube
    // TODO: Now just heating the border
    equation_params.f = [](double x, double y, double z, double t, double u) {
        if (x > 0 && x < 1 && y > 0 && y < 1 && z > 0 && z < 1) {
            return 0.0;
        }
        return 1000.0;
    };

    // Specifying the grid parameters
    // We will have a 103x103x103 grid with 1000 iterations along time
    heat_solver_3d::grid_params_t grid_params(1.02 / 102.0, 60.0 / 1000.0);

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
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    run_solver();
    MPI_Finalize();
    return 0;
}
