#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>

PFNGLWINDOWPOS2IPROC glWindowPos2i;

#define DEFAULT_MIN_TEMP 293.15
#define DEFAULT_MAX_TEMP 2000.0

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

static char **file_paths = NULL;
static int file_count = 0;

static int file_curr_idx = 0;
static struct mapped_results_file_t file_curr_map = {};

static size_t layer_x_start_off = 1;
static size_t layer_x_end_off   = 1;
static size_t layer_y_start_off = 1;
static size_t layer_y_end_off   = 1;
static size_t layer_z_start_off = 1;
static size_t layer_z_end_off   = 1;

static int window_curr_w = 800;
static int window_curr_h = 600;
static int window_draw_info = 1;
static int window_draw_legend = 1;

static int mouse_drag_active = 0;
static int mouse_drag_curr_x = 0;
static int mouse_drag_curr_y = 0;

static int rotation_curr_x = 45;
static int rotation_curr_y = 0;

static float camera_curr_off = 15.0f;

#define PLAYBACK_MAX_FPS 60
static int playback_active = 0;
static int playback_fps_curr = 4;

enum temp_mode_t
{
    TEMP_MODE_FIRST      = 0,
    TEMP_MODE_ABS_LIN    = 0,
    TEMP_MODE_ABS_LOG    = 1,
    TEMP_MODE_MINMAX_LIN = 2,
    TEMP_MODE_MINMAX_LOG = 3,
    TEMP_MODE_LAST       = 3
};
static const char * const temp_mode_names[4] = {
    "abs_lin", "abs_log", "minmax_lin", "minmax_log"
};
static enum temp_mode_t temp_mode_curr = TEMP_MODE_FIRST;

struct draw_state_t
{
    double x_step;
    double y_step;
    double z_step;
    double min_temp;
    double max_temp;
};

void init_color_vec(const struct draw_state_t * const draw_state, const double u,
                    float * const color_vec)
{
    const double temp_scale = draw_state->max_temp - draw_state->min_temp;
    double temp_scale_val = 0.0;
    if (temp_scale > 0.0) {
        temp_scale_val = (u - draw_state->min_temp) / temp_scale;
        if (temp_scale_val > 1.0) {
            temp_scale_val = 1.0;
        } else if (temp_scale_val < 0) {
            temp_scale_val = 0.0;
        }
        if (temp_mode_curr == TEMP_MODE_ABS_LOG ||
            temp_mode_curr == TEMP_MODE_MINMAX_LOG) {
            temp_scale_val = log2(1 + temp_scale_val * 7.0) / 3.0;
        }
    }

    color_vec[0] = (temp_scale_val > 0.5) ? 1.0 : (2.0 * temp_scale_val);
    color_vec[1] = (temp_scale_val < 0.5) ? 0.2 : (0.2 + 1.6 * (temp_scale_val - 0.5));
    color_vec[2] = (temp_scale_val > 0.5) ? (2.0 * (temp_scale_val - 0.5)) : (1.0 - 2.0 * temp_scale_val);
}

static inline size_t get_idx(const size_t i, const size_t j, const size_t k)
{
    return k + file_curr_map.contents->z_dim_size * (file_curr_map.contents->y_dim_size * i + j);
}

void draw_rectangle(const struct draw_state_t * const draw_state,
                    const size_t i, const size_t j, const size_t k,
                    const unsigned dim_1, const unsigned dim_2)
{
    const struct results_file_t * const contents = file_curr_map.contents;
    const unsigned mov[4][3] = {
        {0, 0, 0},
        {dim_1 == 0, dim_1 == 1, dim_1 == 2},
        {dim_1 == 0 || dim_2 == 0, dim_1 == 1 || dim_2 == 1, dim_1 == 2 || dim_2 == 2},
        {dim_2 == 0, dim_2 == 1, dim_2 == 2}
    };
    const double grid_offset[3] = {
        (dim_1 != 0 && dim_2 != 0 && i != layer_x_start_off) ? 0.5 : -0.5,
        (dim_1 != 1 && dim_2 != 1 && j != layer_y_start_off) ? 0.5 : -0.5,
        (dim_1 != 2 && dim_2 != 2 && k != layer_z_start_off) ? 0.5 : -0.5
    };
    const double start_coords[3] = {
        contents->x_start + (i + grid_offset[0]) * draw_state->x_step,
        contents->y_start + (j + grid_offset[1]) * draw_state->y_step,
        contents->z_start + (k + grid_offset[2]) * draw_state->z_step
    };

    const size_t idx = get_idx(i, j, k);
    float color_vec[3];
    init_color_vec(draw_state, contents->grid[idx], color_vec);
    glColor3fv(color_vec);

    for (size_t vi = 0; vi < 4; ++vi) {
        glVertex3f(start_coords[0] + mov[vi][0] * draw_state->x_step,
                   start_coords[1] + mov[vi][1] * draw_state->y_step,
                   start_coords[2] + mov[vi][2] * draw_state->z_step);
    }
}

void update_minmax_temps(struct draw_state_t * const draw_state,
                         const size_t i, const size_t j, const size_t k)
{
    const double u = file_curr_map.contents->grid[get_idx(i, j, k)];
    if (u < draw_state->min_temp) {
        draw_state->min_temp = u;
    }
    if (u > draw_state->max_temp) {
        draw_state->max_temp = u;
    }
}

void init_minmax_temps(struct draw_state_t * const draw_state)
{
    const struct results_file_t * const contents = file_curr_map.contents;

    draw_state->min_temp = DBL_MAX;
    draw_state->max_temp = -DBL_MAX;

    for (size_t i = layer_x_start_off; i < contents->x_dim_size - layer_x_end_off; ++i) {
        for (size_t j = layer_y_start_off; j < contents->y_dim_size - layer_y_end_off; ++j) {
            update_minmax_temps(draw_state, i, j, layer_z_start_off);
            update_minmax_temps(draw_state, i, j, contents->z_dim_size - layer_z_end_off - 1);
        }
    }

    for (size_t i = layer_x_start_off; i < contents->x_dim_size - layer_x_end_off; ++i) {
        for (size_t k = layer_z_start_off; k < contents->z_dim_size - layer_z_end_off; ++k) {
            update_minmax_temps(draw_state, i, layer_y_start_off, k);
            update_minmax_temps(draw_state, i, contents->y_dim_size - layer_y_end_off - 1, k);
        }
    }

    for (size_t j = layer_y_start_off; j < contents->y_dim_size - layer_y_end_off; ++j) {
        for (size_t k = layer_z_start_off; k < contents->z_dim_size - layer_z_end_off; ++k) {
            update_minmax_temps(draw_state, layer_x_start_off, j, k);
            update_minmax_temps(draw_state, contents->x_dim_size - layer_x_end_off - 1, j, k);
        }
    }
}

void draw_rectangles(struct draw_state_t * const draw_state)
{
    glBegin(GL_QUADS);

    const struct results_file_t * const contents = file_curr_map.contents;

    for (size_t i = layer_x_start_off; i < contents->x_dim_size - layer_x_end_off; ++i) {
        for (size_t j = layer_y_start_off; j < contents->y_dim_size - layer_y_end_off; ++j) {
            draw_rectangle(draw_state, i, j, layer_z_start_off, 1, 0);
            draw_rectangle(draw_state, i, j, contents->z_dim_size - layer_z_end_off - 1, 0, 1);
        }
    }

    for (size_t i = layer_x_start_off; i < contents->x_dim_size - layer_x_end_off; ++i) {
        for (size_t k = layer_z_start_off; k < contents->z_dim_size - layer_z_end_off; ++k) {
            draw_rectangle(draw_state, i, layer_y_start_off, k, 0, 2);
            draw_rectangle(draw_state, i, contents->y_dim_size - layer_y_end_off - 1, k, 2, 0);
        }
    }

    for (size_t j = layer_y_start_off; j < contents->y_dim_size - layer_y_end_off; ++j) {
        for (size_t k = layer_z_start_off; k < contents->z_dim_size - layer_z_end_off; ++k) {
            draw_rectangle(draw_state, layer_x_start_off, j, k, 2, 1);
            draw_rectangle(draw_state, contents->x_dim_size - layer_x_end_off - 1, j, k, 1, 2);
        }
    }

    glEnd();
}

void set_camera()
{
    double aspect_ratio = (double)window_curr_w / (double)window_curr_h;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, aspect_ratio, 0.5, 100.0);
    glTranslatef(0.0f, 0.0f, -camera_curr_off);
    glRotatef(rotation_curr_y, 1.0f, 0.0f, 0.0f);
    glRotatef(rotation_curr_x, 0.0f, 1.0f, 0.0f);

    const struct results_file_t * const contents = file_curr_map.contents;
    const float x_mid = (contents->x_start + contents->x_end) / 2.0;
    const float y_mid = (contents->y_start + contents->y_end) / 2.0;
    const float z_mid = (contents->z_start + contents->z_end) / 2.0;
    glTranslatef(-x_mid, -y_mid, -z_mid);

    glMatrixMode(GL_MODELVIEW);
}

void reshape(int w, int h)
{
    window_curr_w = w;
    window_curr_h = h;
    glViewport(0, 0, w, h);
    set_camera();
}

void init_draw_state(struct draw_state_t * const draw_state)
{
    const struct results_file_t * const contents = file_curr_map.contents;
    draw_state->x_step = (contents->x_end - contents->x_start) / (contents->x_dim_size - 1);
    draw_state->y_step = (contents->y_end - contents->y_start) / (contents->y_dim_size - 1);
    draw_state->z_step = (contents->z_end - contents->z_start) / (contents->z_dim_size - 1);
    draw_state->min_temp = DEFAULT_MIN_TEMP;
    draw_state->max_temp = DEFAULT_MAX_TEMP;
    if (temp_mode_curr == TEMP_MODE_MINMAX_LIN ||
        temp_mode_curr == TEMP_MODE_MINMAX_LOG) {
        init_minmax_temps(draw_state);
    }
}

void draw_info(struct draw_state_t * const draw_state)
{
    if (!window_draw_info) {
        return;
    }

    char text_buf[1024];
    snprintf(text_buf, sizeof(text_buf),
             "X: from %.1f to %.1f cm\n"
             "Y: from %.1f to %.1f cm\n"
             "Z: from %.1f to %.1f cm\n"
             "T: from %.2f to %.2f s\n"
             "------------------------------\n"
             "Current X: from %.1f to %.1f cm\n"
             "Current Y: from %.1f to %.1f cm\n"
             "Current Z: from %.1f to %.1f cm\n"
             "Current T: %.2f s\n"
             "------------------------------\n"
             "Temperature Mode: %s\n"
             "Playback State: %s\n"
             "Playback Direction: %s\n"
             "Playback Target FPS: %d\n",
             file_curr_map.contents->x_start,
             file_curr_map.contents->x_end,
             file_curr_map.contents->y_start,
             file_curr_map.contents->y_end,
             file_curr_map.contents->z_start,
             file_curr_map.contents->z_end,
             file_curr_map.contents->t_start,
             file_curr_map.contents->t_end,
             file_curr_map.contents->x_start + layer_x_start_off * draw_state->x_step,
             file_curr_map.contents->x_end - layer_x_end_off * draw_state->x_step,
             file_curr_map.contents->y_start + layer_y_start_off * draw_state->y_step,
             file_curr_map.contents->y_end - layer_y_end_off * draw_state->y_step,
             file_curr_map.contents->z_start + layer_z_start_off * draw_state->z_step,
             file_curr_map.contents->z_end - layer_z_end_off * draw_state->z_step,
             file_curr_map.contents->t,
             temp_mode_names[temp_mode_curr],
             playback_active ? "Running" : "Paused",
             playback_fps_curr >= 0 ? "Forward" : "Backward",
             playback_fps_curr >= 0 ? playback_fps_curr : -playback_fps_curr);
    glColor3f(1.0f, 1.0f, 1.0f);
    glWindowPos2i(1, window_curr_h - 10);
    glutBitmapString(GLUT_BITMAP_8_BY_13, text_buf);
}

void draw_legend(struct draw_state_t * const draw_state)
{
    if (!window_draw_legend) {
        return;
    }

    /* Drawing the gradient */

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, window_curr_w, 0.0, window_curr_h);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    const int bin_width = 20;
    const int bin_height = 20;
    const int bin_count = 9;
    const double bin_temp_step = (draw_state->max_temp - draw_state->min_temp) / (bin_count + 1);
    const int y_start = (window_curr_h  - bin_height * bin_count) / 2;
    const int x_start = window_curr_w - bin_width - 10;

    float color_vec[3];
    init_color_vec(draw_state, draw_state->min_temp, color_vec);
    glColor3fv(color_vec);

    glBegin(GL_QUADS);
    for (size_t i = 0; i < bin_count; ++i) {
        glVertex2i(x_start, y_start + i * bin_height);
        glVertex2i(x_start + bin_width, y_start + i * bin_height);

        init_color_vec(draw_state, draw_state->min_temp + (i + 1) * bin_temp_step, color_vec);
        glColor3fv(color_vec);

        glVertex2i(x_start + bin_width, y_start + (i + 1) * bin_height);
        glVertex2i(x_start, y_start + (i + 1) * bin_height);
    }
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    /* Drawing the labels */

    const int label_count = 8;
    const int label_block_height = (bin_height * bin_count) / (label_count - 1);
    const double label_temp_step = (draw_state->max_temp - draw_state->min_temp) / (label_count - 1);

    char label_buf[32];
    int label_size;

    glColor3f(1.0f, 1.0f, 1.0f);
    for (size_t i = 0; i < label_count; ++i) {
        label_size = snprintf(label_buf, sizeof(label_buf), "%.2fK", draw_state->min_temp + i * label_temp_step);
        glWindowPos2i(x_start - label_size * 8, y_start + i * label_block_height - 6);
        glutBitmapString(GLUT_BITMAP_8_BY_13, label_buf);
    }
}

void display()
{
    struct draw_state_t draw_state;
    init_draw_state(&draw_state);

    glClear(GL_COLOR_BUFFER_BIT);
    set_camera();
    draw_rectangles(&draw_state);
    draw_info(&draw_state);
    draw_legend(&draw_state);
    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
    /* Pause/resume playback */
    if (key == ' ') {
        playback_active = !playback_active;
        if (window_draw_info) {
            glutPostRedisplay();
        }
    /* Change temperature display mode */
    } else if (key == 'm') {
        if (temp_mode_curr == TEMP_MODE_LAST) {
            temp_mode_curr = TEMP_MODE_FIRST;
        } else {
            ++temp_mode_curr;
        }
        glutPostRedisplay();
    /* Strip one layer from x start */
    } else if (key == 'q' &&
               layer_x_start_off + layer_x_end_off < file_curr_map.contents->x_dim_size - 2) {
        ++layer_x_start_off;
        glutPostRedisplay();
    /* Add one layer to x start */
    } else if (key == 'a' && layer_x_start_off != 0) {
        --layer_x_start_off;
        glutPostRedisplay();
    /* Strip one layer from x end */
    } else if (key == 's' &&
               layer_x_start_off + layer_x_end_off < file_curr_map.contents->x_dim_size - 2) {
        ++layer_x_end_off;
        glutPostRedisplay();
    /* Add one layer to x end */
    } else if (key == 'w' && layer_x_end_off != 0) {
        --layer_x_end_off;
        glutPostRedisplay();
    /* Strip one layer from y start */
    } else if (key == 'e' &&
               layer_y_start_off + layer_y_end_off < file_curr_map.contents->y_dim_size - 2) {
        ++layer_y_start_off;
        glutPostRedisplay();
    /* Add one layer to y start */
    } else if (key == 'd' && layer_y_start_off != 0) {
        --layer_y_start_off;
        glutPostRedisplay();
    /* Strip one layer from y end */
    } else if (key == 'f' &&
               layer_y_start_off + layer_y_end_off < file_curr_map.contents->y_dim_size - 2) {
        ++layer_y_end_off;
        glutPostRedisplay();
    /* Add one layer to y end */
    } else if (key == 'r' && layer_y_end_off != 0) {
        --layer_y_end_off;
        glutPostRedisplay();
    /* Strip one layer from z start */
    } else if (key == 't' &&
               layer_z_start_off + layer_z_end_off < file_curr_map.contents->z_dim_size - 2) {
        ++layer_z_start_off;
        glutPostRedisplay();
    /* Add one layer to z start */
    } else if (key == 'g' && layer_z_start_off != 0) {
        --layer_z_start_off;
        glutPostRedisplay();
    /* Strip one layer from z end */
    } else if (key == 'h' &&
               layer_z_start_off + layer_z_end_off < file_curr_map.contents->z_dim_size - 2) {
        ++layer_z_end_off;
        glutPostRedisplay();
    /* Add one layer to z end */
    } else if (key == 'y' && layer_z_end_off != 0) {
        --layer_z_end_off;
        glutPostRedisplay();
    }
}

void load_new_file()
{
    unmap_file(&file_curr_map);
    if (map_file(file_paths[file_curr_idx], &file_curr_map)) {
        exit(1);
    }
    glutPostRedisplay();
}

void special(int key, int x, int y)
{
    /* Increase playback FPS */
    if (key == GLUT_KEY_RIGHT && playback_fps_curr < PLAYBACK_MAX_FPS) {
        ++playback_fps_curr;
        if (window_draw_info) {
            glutPostRedisplay();
        }
    /* Decrease playback FPS */
    } else if (key == GLUT_KEY_LEFT && playback_fps_curr > -PLAYBACK_MAX_FPS) {
        --playback_fps_curr;
        if (window_draw_info) {
            glutPostRedisplay();
        }
    /* Load previous file */
    } else if (key == GLUT_KEY_DOWN && file_curr_idx != 0) {
        --file_curr_idx;
        load_new_file();
    /* Load next file */
    } else if (key == GLUT_KEY_UP && file_curr_idx != file_count - 1) {
        ++file_curr_idx;
        load_new_file();
    /* Load first file */
    } else if (key == GLUT_KEY_HOME) {
        file_curr_idx = 0;
        load_new_file();
    /* Load last file */
    } else if (key == GLUT_KEY_END) {
        file_curr_idx = file_count - 1;
        load_new_file();
    /* Show/hide info */
    } else if (key == GLUT_KEY_F1) {
        window_draw_info = !window_draw_info;
        glutPostRedisplay();
    /* Show/hide legend */
    } else if (key == GLUT_KEY_F2) {
        window_draw_legend = !window_draw_legend;
        glutPostRedisplay();
    }
}

void mouse(int button, int state, int x, int y)
{
    /* Rotate the camera (start) */
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            mouse_drag_active = 1;
            mouse_drag_curr_x = x;
            mouse_drag_curr_y = y;
        } else if (state == GLUT_UP) {
            mouse_drag_active = 0;
        }
    /* Zoom in */
    } else if (button == 3 && state == GLUT_DOWN && camera_curr_off > 0.0f) {
        camera_curr_off -= 0.5f;
        glutPostRedisplay();
    /* Zoom out */
    } else if (button == 4 && state == GLUT_DOWN) {
        camera_curr_off += 0.5f;
        glutPostRedisplay();
    }
}

void motion(int x, int y)
{
    if (!mouse_drag_active) {
        return;
    }

    const int x_d = x - mouse_drag_curr_x;
    const int y_d = y - mouse_drag_curr_y;
    if (!x_d && !y_d) {
        return;
    }
    mouse_drag_curr_x = x;
    mouse_drag_curr_y = y;
    rotation_curr_x += x_d;
    rotation_curr_y += y_d;
    glutPostRedisplay();
}

void timer(int value)
{
    if (playback_active && playback_fps_curr) {
        if (playback_fps_curr > 0 && file_curr_idx != file_count - 1) {
            ++file_curr_idx;
            load_new_file();
        } else if (playback_fps_curr < 0 && file_curr_idx != 0) {
            --file_curr_idx;
            load_new_file();
        }
    }

    unsigned delay = 1000;
    if (!playback_fps_curr) {
        delay /= PLAYBACK_MAX_FPS;
    } else if (playback_fps_curr < 0) {
        delay /= (-playback_fps_curr);
    } else {
        delay /= playback_fps_curr;
    }
    glutTimerFunc(delay, timer, 0);
}

void init_glut(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glWindowPos2i = (PFNGLWINDOWPOS2IPROC)glutGetProcAddress("glWindowPos2i");
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_MULTISAMPLE);
    glutInitWindowPosition(80, 80);
    glutInitWindowSize(window_curr_w, window_curr_h);
    glutCreateWindow("Plasma Torch Visualization");
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(1000 / PLAYBACK_MAX_FPS, timer, 0);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file 0> <file 1> ...\n", argv[0]);
        return 1;
    }

    file_paths = argv + 1;
    file_count = argc - 1;
    /* Will open the first file here */
    file_curr_idx = 0;
    if (map_file(file_paths[file_curr_idx], &file_curr_map)) {
        return 1;
    }

    init_glut(argc, argv);
    glutMainLoop();

    return 0;
}
