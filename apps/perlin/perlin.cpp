#include "perlin.h"
#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"
#include <cstdio>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: ./process output.png");
        return 0;
    }

    Halide::Runtime::Buffer<uint8_t, 2> output(3840, 2160);
    Halide::Tools::BenchmarkResult result = Halide::Tools::benchmark([&output]() {
            perlin(32.0f / 1024, output);
    });
    std::printf("Perlin(%d, %d): %.3fms\n", output.width(), output.height(), result * 1000.0);


    Halide::Tools::convert_and_save_image(output, argv[1]);

    return 0;
}

