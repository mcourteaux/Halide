#include "Halide.h"
#include "halide_benchmark.h"

#ifndef M_PI
#define M_PI 3.14159265358979310000
#endif

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    if (target.arch == Target::WebAssembly) {
        printf("[SKIP] Performance tests are meaningless and/or misleading under WebAssembly interpreter.\n");
        return 0;
    }

    Var x, y;
    const int test_w = 256;
    const int test_h = 256;

    Expr t0 = x / float(test_w);
    Expr t1 = y / float(test_h);
    // To make sure we time mostely the computation of the arctan, and not memory bandwidth,
    // we will compute many arctans per output and sum them. In my testing, GPUs suffer more
    // from bandwith with this test, so we give it more arctangents to compute per output.
    const int test_d = target.has_gpu_feature() ? 1024 : 64;
    RDom rdom{0, test_d};
    Expr off = rdom / float(test_d) - 0.5f;

    float range = -10.0f;
    Func atan_ref{"atan_ref"}, atan2_ref{"atan2_ref"};
    atan_ref(x, y) = sum(atan(-range * t0 + (1 - t0) * range + off));
    atan2_ref(x, y) = sum(atan2(-range * t0 + (1 - t0) * range + off, -range * t1 + (1 - t1) * range));

    Var xo, xi;
    Var yo, yi;
    if (target.has_gpu_feature()) {
        atan_ref.never_partition_all();
        atan2_ref.never_partition_all();
        atan_ref.gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::ShiftInwards);
        atan2_ref.gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::ShiftInwards);
    } else {
        atan_ref.vectorize(x, 8);
        atan2_ref.vectorize(x, 8);
    }

    double scale = 1e9 / (double(test_w) * (test_h * test_d));
    Buffer<float> atan_out(test_w, test_h);
    Buffer<float> atan2_out(test_w, test_h);
    atan_ref.compile_jit();
    atan2_ref.compile_jit();
    // clang-format off
    double t_atan  = scale * benchmark([&]() {  atan_ref.realize( atan_out);  atan_out.device_sync(); });
    double t_atan2 = scale * benchmark([&]() { atan2_ref.realize(atan2_out); atan2_out.device_sync(); });
    // clang-format on

    struct Prec {
        ApproximationPrecision precision;
        const char *name;
        double atan_time{0.0f};
        double atan2_time{0.0f};
    } precisions_to_test[] = {
        {ApproximationPrecision::MULPE_Poly2, "Poly2"},
        {ApproximationPrecision::MULPE_Poly3, "Poly3"},
        {ApproximationPrecision::MULPE_Poly4, "Poly4"},
        {ApproximationPrecision::MULPE_Poly5, "Poly5"},
        {ApproximationPrecision::MULPE_Poly6, "Poly6"},
        {ApproximationPrecision::MULPE_Poly7, "Poly7"},
        {ApproximationPrecision::MULPE_Poly8, "Poly8"},
    };

    for (Prec &precision : precisions_to_test) {
        Func atan_f{"fast_atan"}, atan2_f{"fast_atan2"};

        atan_f(x, y) = sum(fast_atan(-range * t0 + (1 - t0) * range + off, precision.precision));
        atan2_f(x, y) = sum(fast_atan2(-range * t0 + (1 - t0) * range + off,
                                       -range * t1 + (1 - t1) * range, precision.precision));

        if (target.has_gpu_feature()) {
            atan_f.never_partition_all();
            atan2_f.never_partition_all();
            atan_f.gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::ShiftInwards);
            atan2_f.gpu_tile(x, y, xo, yo, xi, yi, 16, 16, TailStrategy::ShiftInwards);
        } else {
            atan_f.vectorize(x, 8);
            atan2_f.vectorize(x, 8);
        }

        atan_f.compile_jit();
        atan2_f.compile_jit();
        // clang-format off
        double t_fast_atan  = scale * benchmark([&]() {  atan_f.realize( atan_out);  atan_out.device_sync(); });
        double t_fast_atan2 = scale * benchmark([&]() { atan2_f.realize(atan2_out); atan2_out.device_sync(); });
        // clang-format on
        precision.atan_time = t_fast_atan;
        precision.atan2_time = t_fast_atan2;
    }

    printf("              atan: %f ns per atan\n", t_atan);
    for (const Prec &precision : precisions_to_test) {
        printf(" fast_atan (%s): %f ns per atan (%4.1f%% faster)  [per invokation: %f ms]\n",
               precision.name, precision.atan_time, 100.0f * (1.0f - precision.atan_time / t_atan),
               precision.atan_time / scale * 1e3);
    }
    printf("\n");
    printf("              atan2: %f ns per atan2\n", t_atan2);
    for (const Prec &precision : precisions_to_test) {
        printf(" fast_atan2 (%s): %f ns per atan2 (%4.1f%% faster)  [per invokation: %f ms]\n",
               precision.name, precision.atan2_time, 100.0f * (1.0f - precision.atan2_time / t_atan2),
               precision.atan2_time / scale * 1e3);
    }

    int num_passed = 0;
    int num_tests = 0;
    for (const Prec &precision : precisions_to_test) {
        num_tests += 2;
        if (t_atan < precision.atan_time) {
            printf("fast_atan is not faster than atan\n");
        } else {
            num_passed++;
        }
        if (t_atan2 < precision.atan2_time) {
            printf("fast_atan2 is not faster than atan2\n");
        } else {
            num_passed++;
        }
    }

    if (num_passed < num_tests) {
        printf("Not all measurements were faster for the fast variants of the atan/atan2 funcions.\n");
        return 1;
    }

    printf("Success!\n");
    return 0;
}