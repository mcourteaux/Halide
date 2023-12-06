#include "Halide.h"

#include <algorithm>

namespace {

using namespace Halide;

class PerlinGenerator : public Halide::Generator<PerlinGenerator> {
public:
    Var x{"x"}, y{"y"}, c{"c"};

    Input<float> scale{"scale"};
    Output<Buffer<uint8_t, 2>> output{"output"};

    Func random_angle{"random_angle"};
    Func random_gradient{"random_gradient"};

    Func dot00{"dot_00"};
    Func dot01{"dot_01"};
    Func dot10{"dot_10"};
    Func dot11{"dot_11"};
    Func perlin{"perlin"};

    Expr fade(Expr t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    Expr dot_grid_gradient(Expr ix, Expr iy, Expr x, Expr y) {
        Expr grad_x = random_gradient(ix, iy, 0);
        Expr grad_y = random_gradient(ix, iy, 1);

        Expr dx = x - cast<float>(ix);
        Expr dy = y - cast<float>(iy);

        return dx * grad_x + dy * grad_y;
    }

    void generate() {
        add_requirement(scale >= 0.0f, "scale must be greater than 0");

        {
            Expr seed = fract(sin(x * 12.9898f + y * 4.1414f) * 43758.5453f);
            Expr random = seed * (3.14159265f * 2.0f);  // in [0, 2*Pi]
            random_angle(x, y) = random;
        }

        random_gradient(x, y, c) = select(c == 0, fast_cos(random_angle(x, y)), fast_sin(random_angle(x, y)));

        {
            Expr scaled_x = scale * x;
            Expr scaled_y = scale * y;

            Expr x0 = floor(scaled_x);
            Expr x0i = cast<int32_t>(scaled_x);
            Expr x1i = x0i + 1;
            Expr y0 = floor(scaled_y);
            Expr y0i = cast<int32_t>(scaled_y);
            Expr y1i = y0i + 1;

            Expr sx = fade(scaled_x - x0);
            Expr sy = fade(scaled_y - y0);

            dot00(x, y) = dot_grid_gradient(x0i, y0i, scaled_x, scaled_y);
            dot01(x, y) = dot_grid_gradient(x1i, y0i, scaled_x, scaled_y);
            Expr ix0 = lerp(dot00(x, y), dot01(x, y), sx);

            dot10(x, y) = dot_grid_gradient(x0i, y1i, scaled_x, scaled_y);
            dot11(x, y) = dot_grid_gradient(x1i, y1i, scaled_x, scaled_y);
            Expr ix1 = lerp(dot10(x, y), dot11(x, y), sx);

            perlin(x, y) = lerp(ix0, ix1, sy);
        }

        output(x, y) = cast<uint8_t>(round(255.0f * (perlin(x, y) * 0.5f + 0.5f)));

        output.dim(0).set_min(0);
        output.dim(1).set_min(0);
    }

    void schedule_cpu() {
        Var xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

        output
            .compute_root()
            //.tile(x, y, xo, yo, xi, yi, 8, 8)
            .split(y, yo, yi, 32)
            .vectorize(x, 8)
            .parallel(yo)
            //.vectorize(xi)
            ;

        perlin
            .compute_at(output, x)
            ;

        random_gradient
            .compute_at(output, yo)
            .reorder(c, x, y)
            .bound(c, 0, 2)
            .unroll(c)
            .partition(x, Partition::Never)
            //.vectorize(x, 8, TailStrategy::ShiftInwards)
            ;

        random_angle
            .compute_at(random_gradient, x)
            .partition(x, Partition::Never)
            //.vectorize(x, 8)
            ;
    }

    void schedule_gpu() {
        Var xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

        output
            .compute_root()
            .gpu_tile(x, y, xo, yo, xi, yi, 256, 8, TailStrategy::ShiftInwards)
            .partition(xi, Partition::Never)
            .partition(yi, Partition::Never)
            .partition(xo, Partition::Never)
            .partition(yo, Partition::Never)
            .vectorize(xi, 8)
            ;

        auto schedule_dot = [&](Func dot) {
            dot
                .compute_at(output, xi)
                .vectorize(x)
                ;
        };

        schedule_dot(dot00);
        schedule_dot(dot01);
        schedule_dot(dot10);
        schedule_dot(dot11);
        dot01.compute_with(dot00, x);
        dot10.compute_with(dot00, x);
        dot11.compute_with(dot00, x);

        random_gradient
            .compute_root()
            .reorder_storage(c, x, y)
            .reorder(c, x, y)
            .gpu_tile(x, y, xo, yo, xi, yi, 64, 8, TailStrategy::ShiftInwards)
            .partition(xi, Partition::Never)
            .partition(yi, Partition::Never)
            .partition(xo, Partition::Never)
            .partition(yo, Partition::Never)
            .unroll(c)
            ;
    }

    void schedule() {
        if (get_target().has_gpu_feature()) {
            std::printf("Schedule for GPU\n");
            schedule_gpu();
        } else {
            std::printf("Schedule for CPU\n");
            schedule_cpu();
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(PerlinGenerator, perlin)
