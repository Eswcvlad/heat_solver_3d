#ifndef LAYER_HPP
#define LAYER_HPP

#include <array>
#include <vector>
#include <ostream>

namespace heat_solver_3d
{

class layer_t
{
public:
    layer_t(const size_t x_dims, const size_t y_dims, const size_t z_dims)
    {
        adjust_dims(x_dims, y_dims, z_dims);
    }

    explicit layer_t(const std::array<size_t, 3> &dims)
    {
        adjust_dims(dims);
    }

    explicit layer_t(const size_t dims[3])
    {
        adjust_dims(dims);
    }

    layer_t()
    {
        adjust_dims(0, 0, 0);
    }

    void adjust_dims(const size_t x_dims, const size_t y_dims, const size_t z_dims)
    {
        _dims[0] = x_dims;
        _dims[1] = y_dims;
        _dims[2] = z_dims;
        _layer_buf.resize(x_dims * y_dims * z_dims);
    }

    void adjust_dims(const std::array<size_t, 3> &dims)
    {
        adjust_dims(dims[0], dims[1], dims[2]);
    }

    void adjust_dims(const size_t dims[3])
    {
        adjust_dims(dims[0], dims[1], dims[2]);
    }

    size_t x_dims() const
    {
        return _dims[0];
    }

    size_t y_dims() const
    {
        return _dims[1];
    }

    size_t z_dims() const
    {
        return _dims[2];
    }

    size_t size() const
    {
        return _dims[0] * _dims[1] * _dims[2];
    }

    double * data()
    {
        return _layer_buf.data();
    }

    const double * data() const
    {
        return _layer_buf.data();
    }

    inline double & operator()(const size_t i, const size_t j, const size_t k)
    {
        return _layer_buf[k + _dims[2] * (_dims[1] * i + j)];
    }

    inline double & operator()(const std::array<size_t, 3> &coords)
    {
        return (*this)(coords[0], coords[1], coords[2]);
    }

    inline double operator()(const size_t i, const size_t j, const size_t k) const
    {
        return _layer_buf[k + _dims[2] * (_dims[1] * i + j)];
    }

    inline double operator()(const std::array<size_t, 3> &coords) const
    {
        return (*this)(coords[0], coords[1], coords[2]);
    }

    const double * get_z_row_ptr(const size_t i, const size_t j) const
    {
        return &_layer_buf[_dims[2] * (_dims[1] * i + j)];
    }

    void swap(layer_t &other)
    {
        _dims.swap(other._dims);
        _layer_buf.swap(other._layer_buf);
    }

    /*
     * Saves in binary, expects a binary stream
     * Format:
     * x_dims y_dims z_dims - optional header
     * layer_buf - the data itself
     */
    void save(std::ostream &os, const bool write_header = true) const
    {
        if (write_header) {
            for (const size_t dim : _dims) {
                uint64_t tmp_dim = static_cast<uint64_t>(dim);
                os.write(reinterpret_cast<const char *>(&tmp_dim),
                         sizeof(tmp_dim));
            }
        }
        os.write(reinterpret_cast<const char *>(_layer_buf.data()),
                 _layer_buf.size() * sizeof(double));
    }

private:
    std::array<size_t, 3> _dims;
    std::vector<double> _layer_buf;
};  // class layer_t

}   // namespace heat_solver_3d

#endif  // LAYER_HPP
