#include "Halide.h"

using namespace Halide;

namespace {

template<typename T>
Halide::Expr is_interleaved(const T &p, int channels = 3) {
    return p.dim(0).stride() == channels && p.dim(2).stride() == 1 && p.dim(2).extent() == channels;
}

template<typename T>
Halide::Expr is_planar(const T &p, int channels = 3) {
    return p.dim(0).stride() == 1 && p.dim(2).extent() == channels;
}

Var x("x"), y("y"), c("c");

Func blur2x2(const Target &target, Func input, Expr width, Expr height) {
    Func input_clamped =
        Halide::BoundaryConditions::repeat_edge(input, {{0, width}, {0, height}});

    Func blur("blur2x2");
    blur(x, y, c) =
        (input_clamped(x - 1, y, c) + input_clamped(x + 1, y, c) +
         input_clamped(x, y - 1, c) + input_clamped(x, y + 1, c)) /
        4.0f;

#if 0
    // Unset default constraints so that specialization works.
    input.output_buffer().dim(0).set_stride(Expr());
    blur.output_buffer().dim(0).set_stride(Expr());

    // Add specialization for input and output buffers that are both planar.
    blur.specialize(is_planar(input.output_buffer()) && is_planar(blur.output_buffer()))
        .vectorize(x, target.natural_vector_size<float>());

    // Add specialization for input and output buffers that are both interleaved.
    blur.specialize(is_interleaved(input.output_buffer()) && is_interleaved(blur.output_buffer()))
        .reorder(c, x, y)
        .vectorize(c);
#endif

    return blur;
}

}  // namespace

HALIDE_REGISTER_G2(
    blur2x2,     // actual C++ fn
    blur2x2_g2,  // build-system name
    Target(),
    Input("input", Float(32), 3),
    Input("width", Int(32)),
    Input("height", Int(32)),
    Output("output", Float(32), 3))
