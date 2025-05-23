#include "HalideRuntime.h"
#include "cpu_features.h"
#include "printer.h"
#include "scoped_mutex_lock.h"

namespace Halide {
namespace Runtime {
namespace Internal {
WEAK halide_can_use_target_features_t custom_can_use_target_features = halide_default_can_use_target_features;

WEAK uint64_t halide_cpu_features_storage[sizeof(CpuFeatures) / sizeof(uint64_t)] = {0};
WEAK bool halide_cpu_features_initialized = false;
WEAK halide_mutex halide_cpu_features_initialized_lock;

}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide

extern "C" {

WEAK halide_can_use_target_features_t halide_set_custom_can_use_target_features(halide_can_use_target_features_t fn) {
    halide_can_use_target_features_t result = custom_can_use_target_features;
    custom_can_use_target_features = fn;
    return result;
}

WEAK int halide_can_use_target_features(int count, const uint64_t *features) {
    return (*custom_can_use_target_features)(count, features);
}

WEAK int halide_default_can_use_target_features(int count, const uint64_t *features) {
    // cpu features should never change, so call once and cache.
    // Note that since CpuFeatures has a (trivial) ctor, compilers may insert guards
    // for threadsafe initialization (per C++11); this can fail at link time
    // on some systems (MSVC) because our runtime is a special beast. We'll
    // work around this by using a sentinel for the initialization flag and
    // some horribleness with memcpy (which we can do since CpuFeatures is still POD).
    {
        ScopedMutexLock lock(&halide_cpu_features_initialized_lock);

        static_assert(sizeof(halide_cpu_features_storage) == sizeof(CpuFeatures), "CpuFeatures Mismatch");
        if (!halide_cpu_features_initialized) {
            CpuFeatures tmp;
            int error = halide_get_cpu_features(&tmp);
            halide_abort_if_false(nullptr, error == halide_error_code_success);
            memcpy(&halide_cpu_features_storage, &tmp, sizeof(tmp));
            halide_cpu_features_initialized = true;
        }
    }

    if (count != cpu_feature_mask_size) {
        // This should not happen unless our runtime is out of sync with the rest of libHalide.
#ifdef DEBUG_RUNTIME
        debug(nullptr) << "count " << count << " cpu_feature_mask_size " << cpu_feature_mask_size << "\n";
#endif
        halide_error(nullptr, "Internal error: wrong structure size passed to halide_can_use_target_features()\n");
    }
    const CpuFeatures *cpu_features = reinterpret_cast<const CpuFeatures *>(&halide_cpu_features_storage[0]);
    for (int i = 0; i < cpu_feature_mask_size; ++i) {
        uint64_t m;
        if ((m = (features[i] & cpu_features->known[i])) != 0) {
            if ((m & cpu_features->available[i]) != m) {
                return 0;
            }
        }
    }

    return 1;
}
}
