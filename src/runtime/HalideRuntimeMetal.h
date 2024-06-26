#ifndef HALIDE_HALIDERUNTIMEMETAL_H
#define HALIDE_HALIDERUNTIMEMETAL_H

// Don't include HalideRuntime.h if the contents of it were already pasted into a generated header above this one
#ifndef HALIDE_HALIDERUNTIME_H

#include "HalideRuntime.h"

#endif

#ifdef __cplusplus
extern "C" {
#endif

/** \file
 *  Routines specific to the Halide Metal runtime.
 */

#define HALIDE_RUNTIME_METAL

extern const struct halide_device_interface_t *halide_metal_device_interface();

/** These are forward declared here to allow clients to override the
 *  Halide Metal runtime. Do not call them. */
// @{
extern int halide_metal_initialize_kernels(void *user_context, void **state_ptr,
                                           const char *src, int size);
void halide_metal_finalize_kernels(void *user_context, void *state_ptr);

extern int halide_metal_run(void *user_context,
                            void *state_ptr,
                            const char *entry_name,
                            int blocksX, int blocksY, int blocksZ,
                            int threadsX, int threadsY, int threadsZ,
                            int shared_mem_bytes,
                            struct halide_type_t arg_types[],
                            void *args[],
                            int8_t arg_is_buffer[]);
// @}

/** Set the underlying MTLBuffer for a halide_buffer_t. This memory should be
 * allocated using newBufferWithLength:options or similar and must
 * have an extent large enough to cover that specified by the halide_buffer_t
 * extent fields. The dev field of the halide_buffer_t must be NULL when this
 * routine is called. This call can fail due to running out of memory
 * or being passed an invalid buffer. The device and host dirty bits
 * are left unmodified. */
extern int halide_metal_wrap_buffer(void *user_context, struct halide_buffer_t *buf, uint64_t buffer);

/** Disconnect a halide_buffer_t from the memory it was previously
 * wrapped around. Should only be called for a halide_buffer_t that
 * halide_metal_wrap_buffer was previously called on. Frees any
 * storage associated with the binding of the halide_buffer_t and the
 * buffer, but does not free the MTLBuffer. The dev field of the
 * halide_buffer_t will be NULL on return.
 */
extern int halide_metal_detach_buffer(void *user_context, struct halide_buffer_t *buf);

/** Return the underlying MTLBuffer for a halide_buffer_t. This buffer must be
 * valid on an Metal device, or not have any associated device
 * memory. If there is no device memory (dev field is NULL), this
 * returns 0.
 */
extern uintptr_t halide_metal_get_buffer(void *user_context, struct halide_buffer_t *buf);

/** Returns the offset associated with the Metal Buffer allocation via device_crop or device_slice. */
extern uint64_t halide_metal_get_crop_offset(void *user_context, struct halide_buffer_t *buf);

struct halide_metal_device;
struct halide_metal_command_queue;
struct halide_metal_command_buffer;

/** This prototype is exported as applications will typically need to
 * replace it to get Halide filters to execute on the same device and
 * command queue used for other purposes. The halide_metal_device is an
 * id \<MTLDevice\> and halide_metal_command_queue is an id \<MTLCommandQueue\>.
 * No reference counting is done by Halide on these objects. They must remain
 * valid until all off the following are true:
 * - A balancing halide_metal_release_context has occurred for each
 *     halide_metal_acquire_context which returned the device/queue
 * - All Halide filters using the context information have completed
 * - All halide_buffer_t objects on the device have had
 *     halide_device_free called or have been detached via
 *     halide_metal_detach_buffer.
 * - halide_device_release has been called on the interface returned from
 *     halide_metal_device_interface(). (This releases the programs on the context.)
 */
extern int halide_metal_acquire_context(void *user_context, struct halide_metal_device **device_ret,
                                        struct halide_metal_command_queue **queue_ret, bool create);

/** This call balances each successful halide_metal_acquire_context call.
 * If halide_metal_acquire_context is replaced, this routine must be replaced
 * as well.
 */
extern int halide_metal_release_context(void *user_context);

/** This function is called as part of the callback when a Metal command buffer completes.
 * The return value, if not halide_error_code_success, will be stashed in Metal runtime and returned
 * to the next call into the runtime, and the error string will be saved as well.
 * The error string will be freed by the caller. The return value must be a valid Halide error code.
 * This is called from the Metal driver, and thus:
 * - Any user_context must be preserved between the call to halide_metal_run and the corresponding callback
 * - The function must be thread-safe
 */
extern int halide_metal_command_buffer_completion_handler(void *user_context, struct halide_metal_command_buffer *buffer,
                                                          char **returned_error_string);

#ifdef __cplusplus
}  // End extern "C"
#endif

#endif  // HALIDE_HALIDERUNTIMEMETAL_H
