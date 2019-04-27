#ifndef HANDLER_HPP
#define HANDLER_HPP

namespace heat_solver_3d
{

class solver_t;

struct handler_t
{
    // Returns whether the calculations should be continued or not
    // Is called at the start of every t step
    virtual bool handle(solver_t &solver) = 0;

    virtual ~handler_t()
    {
    }
};  // struct handler_t

}   // namespace heat_solver_3d

#endif  // HANDLER_HPP
