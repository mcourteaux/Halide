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

Func blur2x2(Func input, Expr width, Expr height) {
    Func input_clamped =
        Halide::BoundaryConditions::repeat_edge(input, {{0, width}, {0, height}});

    Func blur("blur2x2");
    blur(x, y, c) =
        (input_clamped(x - 1, y, c) + input_clamped(x + 1, y, c) +
         input_clamped(x, y - 1, c) + input_clamped(x, y + 1, c)) /
        4.0f;

    return blur;
}

Func blur2x2_scheduled(const Target &target, ImageParam input, Expr width, Expr height) {
    Func blur = blur2x2(input, width, height);


    // Unset default constraints so that specialization works.
    input.dim(0).set_stride(Expr());
    blur.output_buffer().dim(0).set_stride(Expr());

    // Add specialization for input and output buffers that are both planar.
    blur.specialize(is_planar(input) && is_planar(blur.output_buffer()))
        .vectorize(x, target.natural_vector_size<float>());

    // Add specialization for input and output buffers that are both interleaved.
    blur.specialize(is_interleaved(input) && is_interleaved(blur.output_buffer()))
        .reorder(c, x, y)
        .vectorize(c);

    return blur;
}

}  // namespace

HALIDE_REGISTER_G2(
    blur2x2_scheduled,     // actual C++ fn
    blur2x2_g2,  // build-system name
    Target(),
    Input("input", Float(32), 3),
    Input("width", Int(32)),
    Input("height", Int(32)),
    Output("output", Float(32), 3))
