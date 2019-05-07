#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#define HEADER_SIZE 9 * sizeof(double) + 4 * sizeof(uint64_t)

#pragma pack(push, 1)
struct results_file_t
{
    double x_start;
    double x_end;
    double y_start;
    double y_end;
    double z_start;
    double z_end;
    double t_start;
    double t_end;
    double t;
    uint64_t x_dim_size;
    uint64_t y_dim_size;
    uint64_t z_dim_size;
    uint64_t t_dim_size;
    double grid[];
};
#pragma pack(pop)

struct mapped_results_file_t
{
    struct results_file_t *contents;
    off_t total_size;
    int fd;
};

void unmap_file(struct mapped_results_file_t *out)
{
    if (out->contents) {
        munmap(out->contents, out->total_size);
    }
    if (out->fd != -1) {
        close(out->fd);
    }
}

int check_file_format(const struct mapped_results_file_t * const results_file)
{
    if (results_file->total_size <= HEADER_SIZE) {
        return -1;
    }

    const struct results_file_t * const data = results_file->contents;
    const uint64_t grid_size = data->x_dim_size * data->y_dim_size * data->z_dim_size;
    if (results_file->total_size != HEADER_SIZE + grid_size) {
        return -1;
    }

    if ((data->x_start < data->x_end) &&
        (data->y_start < data->y_end) &&
        (data->z_start < data->z_end) &&
        (data->t_start < data->t_end) &&
        (data->t_start <= data->t && data->t <= data->t_end) &&
        (data->x_dim_size > 0) &&
        (data->y_dim_size > 0) &&
        (data->z_dim_size > 0) &&
        (data->t_dim_size > 0)) {
        return 0;
    }

    return -1;
}

int map_file(const char * const file_path, struct mapped_results_file_t *out)
{
    memset(out, 0, sizeof(*out));

    out->fd = open(file_path, O_RDONLY);
    if (out->fd == -1) {
        perror("open");
        return -1;
    }

    struct stat sb;
    if (fstat(out->fd, &sb) == -1) {
        perror("fstat");
        unmap_file(out);
        return -1;
    }
    out->total_size = sb.st_size;

    out->contents = mmap(NULL, out->total_size, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->contents == MAP_FAILED) {
        perror("mmap");
        unmap_file(out);
        return -1;
    }

    if (!check_file_format(out)) {
        fprintf(stderr, "check_file_format: Invalid file format");
        unmap_file(out);
        return -1;
    }

    return 0;
}

void print_header(const char * const file_path,
                  const struct results_file_t * const contents)
{
    printf("Results Reader v1.0.0\n"
           "---------------------\n"
           "Currently reading %s\n"
           "Iterations: %"PRIu64"\n"
           "Grid: %"PRIu64"x%"PRIu64"x%"PRIu64"\n"
           "X: from %f to %f\n"
           "Y: from %f to %f\n"
           "Z: from %f to %f\n"
           "T: from %f to %f, currently at %f\n"
           "Type commands in the console below\n"
           "Type 'h' or 'help' to show a list of available commands!\n",
           file_path,
           contents->t_dim_size / 3,
           contents->x_dim_size, contents->y_dim_size, contents->z_dim_size,
           contents->x_start, contents->x_end,
           contents->y_start, contents->y_end,
           contents->z_start, contents->z_end,
           contents->t_start, contents->t_end, contents->t);
}

void print_command_list()
{
    puts("Format: command - description\n"
         "Arguments are written in brackets, ex. 'pgrid <i> <j> <k>'\n"
         "h                 - prints this message\n"
         "phead             - prints the header message again\n"
         "pgrid <i> <j> <k> - prints the grid value using the grid\n"
         "                    coordinates (i,j,k), start from 0\n"
         "pcoor <x> <y> <z> - prints the grid value, that is on the grid\n"
         "                    and is closest to the (x,y,z) point\n"
         "exit              - exit the program");
}

uint64_t coor_to_grid(const double start, const double end,
                      const uint64_t dim_size, const double point)
{
    if (point <= start) {
        return 0;
    } else if (point >= end) {
        return dim_size - 1;
    } else {
        const double step = (end - start) / (dim_size - 1);
        return round((point - start) / step);
    }
}

void get_closest_grid_point(const struct results_file_t * const contents,
                            const double x, const double y, const double z,
                            uint64_t * const i, uint64_t * const j, uint64_t * const k)
{
    *i = coor_to_grid(contents->x_start, contents->x_end, contents->x_dim_size, x);
    *j = coor_to_grid(contents->y_start, contents->y_end, contents->y_dim_size, y);
    *k = coor_to_grid(contents->z_start, contents->z_end, contents->z_dim_size, z);
}

void print_val_at(const struct results_file_t * const contents,
                  const uint64_t i, const uint64_t j, const uint64_t k)
{
    if (i >= contents->x_dim_size ||
        j >= contents->y_dim_size ||
        k >= contents->z_dim_size) {
        puts("Out of bounds!");
        return;
    }

    const double x_step = (contents->x_end - contents->x_start) / (contents->x_dim_size - 1);
    const double y_step = (contents->y_end - contents->y_start) / (contents->y_dim_size - 1);
    const double z_step = (contents->z_end - contents->z_start) / (contents->z_dim_size - 1);
    const double x = contents->x_start + i * x_step;
    const double y = contents->y_start + j * y_step;
    const double z = contents->z_start + k * z_step;
    const double val = contents->grid[k + contents->z_dim_size * (contents->y_dim_size * i + j)];
    printf("(%f, %f, %f) -> %f\n", x, y, z, val);
}

void process_commands(const char * const file_path,
                      const struct results_file_t * const contents)
{
    print_header(file_path, contents);

    uint64_t i, j, k;
    double x, y, z;
    char buf[128];
    while (1) {
        fputs("> ", stdout);
        char *res = fgets(buf, sizeof(buf), stdin);
        // if EOF, print \n for better terminal output
        if (!res) {
            putchar('\n');
            break;
        } else if (!strcmp(buf, "exit\n")) {
            break;
        } else if (!strcmp(buf, "h\n") || !strcmp(buf, "help\n")) {
            print_command_list();
        } else if (!strcmp(buf, "phead\n")) {
            print_header(file_path, contents);
        } else if (sscanf(buf, "pgrid %"SCNu64" %"SCNu64" %"SCNu64"\n", &i, &j, &k) == 3) {
            print_val_at(contents, i, j, k);
        } else if (sscanf(buf, "pcoor %lf %lf %lf\n", &x, &y, &z) == 3) {
            get_closest_grid_point(contents, x, y, z, &i, &j, &k);
            print_val_at(contents, i, j, k);
        } else if (strcmp(buf, "\n")) {
            puts("Invalid command!");
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <results file path>\n", argv[0]);
        return 1;
    }

    struct mapped_results_file_t results_file;
    if (map_file(argv[1], &results_file)) {
        return 1;
    }
    process_commands(argv[1], results_file.contents);
    unmap_file(&results_file);

    return 0;
}
