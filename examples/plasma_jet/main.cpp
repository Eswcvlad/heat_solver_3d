#define _USE_MATH_DEFINES

#include <heat_solver_3d/solver.hpp>

#include <mpi.h>

#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

// Creating the handler, that will log to stderr the current step and time
class plasma_jet_handler_t : public heat_solver_3d::handler_t
{
public:
    plasma_jet_handler_t(const int proc_rank, const std::string file_path_prefix)
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
        if (solver.current_step() % 10) {
            return;
        }

        std::stringstream file_path;
        file_path << _file_path_prefix
                  << '.' << std::setfill('0') << std::setw(5)
                  << solver.current_step();

        std::ofstream os;
        if (!_proc_rank) {
            os.open(file_path.str(), std::ios_base::out   |
                                     std::ios_base::trunc |
                                     std::ios_base::binary);
        }
        solver.save_full_grid_state(os);
    }

    const int _proc_rank;
    const std::string _file_path_prefix;
};

double norm_pdf(const double sigma_sq, const double off_sq)
{
    const double coef = 1.0 / sqrt(2.0 * M_PI * sigma_sq);
    return coef * exp(-off_sq / (2.0 * sigma_sq));
}

double jet_x_move(double x)
{
    const double p = 120.0;
    x -= p / 4.0;
    const double mul = std::floor(2.0 * x / p + 0.5);
    const double wave_val = 4.0 / p * (x - p / 2.0 * mul) * std::pow(-1.0, mul);
    return 5.0 + 4.5 * wave_val;
}

double jet_y_move(const double x)
{
    const auto f = [](const double x) {
        return x - std::sin(x);
    };
    const double pi_2 = 2.0 * M_PI;
    const double w = 60.0;
    const double h = 6.0 / 2.0;
    const double x_scale = pi_2 / w * x - M_PI;
    const double y_scale = h / pi_2;
    return 2.0 + y_scale * f(f(x_scale));
}

void run_solver()
{
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    // Setting up our handler
    plasma_jet_handler_t handler(proc_rank, "plasma-jet-res.bin");

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
    // It is quasilinear
    equation_params.is_quasilinear = true;
    // Specifying the space and time limits
    // Space will be [0, 10] in cm, except for z
    // Time: [0, 180]
    equation_params.x_limits = {{0.0, 10.0}};
    equation_params.y_limits = {{0.0, 10.0}};
    equation_params.z_limits = {{0.0, 5.0}};
    equation_params.t_limits = {{0.0, 180.0}};

    // Cube is from Al2O3
    auto diffusivity = [](double x, double y, double z, double t, double u) {
        return 19.272 / u + 4.1579e8 / ((u * u) * (u * u));
    };
    equation_params.g[0] = diffusivity;
    equation_params.g[1] = diffusivity;
    equation_params.g[2] = diffusivity;

    // Will be solving in K
    equation_params.u0 = [](double x, double y, double z, double t) {
        return 293.15;
    };
    auto def_gamma = [](double a, double b, double t) {
        return 293.15;
    };
    for (size_t i = 0; i < 5; ++i) {
        equation_params.gamma[i] = def_gamma;
    }
    // Heat comes from this part
    // Running for 3 minutes
    equation_params.gamma[5] = [](double x, double y, double t) {
        const double sigma_sq = 3.0;
        const double power_mul = (2000.0 - 293.15) / norm_pdf(sigma_sq, 0.0);
        const double x_c = jet_x_move(t);
        const double y_c = jet_y_move(t);
        const double dist_sq = (x - x_c) * (x - x_c) + (y - y_c) * (y - y_c);
        return 293.15 + power_mul * norm_pdf(sigma_sq, dist_sq);
    };

    // Specifying the grid parameters
    // We will have a 101x101x51 grid with 7200 iterations along time
    heat_solver_3d::grid_params_t grid_params(10.0 / 100.0, 180.0 / 7200.0);

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
