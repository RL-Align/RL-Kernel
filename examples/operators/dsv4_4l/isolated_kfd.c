// SPDX-License-Identifier: Apache-2.0
// Keep the real KFD aperture query consistent with the container's render nodes.
// No model, tensor, arithmetic, queue or memory operation is substituted.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <linux/kfd_ioctl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

static uint32_t allowed[64];
static size_t allowed_count;
static dev_t kfd_device;
static int (*next_ioctl)(int, unsigned long, ...);

static int permitted(uint32_t id) {
    for (size_t i = 0; i < allowed_count; ++i)
        if (allowed[i] == id) return 1;
    return 0;
}

__attribute__((constructor)) static void initialize(void) {
    struct stat st;
    if (!stat("/dev/kfd", &st) && S_ISCHR(st.st_mode)) kfd_device = st.st_rdev;
    next_ioctl = dlsym(RTLD_NEXT, "ioctl");
    const char *value = getenv("RLK_ALLOWED_KFD_GPU_IDS");
    if (!next_ioctl || !value || !*value) {
        fputs("RLK_KFD_ISOLATION: missing real ioctl or allowed GPU IDs\n", stderr);
        abort();
    }
    char *list = strdup(value), *cursor = list, *item;
    while ((item = strsep(&cursor, ","))) {
        char *end;
        errno = 0;
        unsigned long id = strtoul(item, &end, 0);
        if (errno || !*item || *end || !id || id > UINT32_MAX ||
            allowed_count == 64 || permitted((uint32_t)id)) abort();
        allowed[allowed_count++] = (uint32_t)id;
    }
    free(list);
}

static uint32_t filter(struct kfd_process_device_apertures *items, uint32_t count) {
    uint32_t kept = 0;
    for (uint32_t i = 0; i < count; ++i)
        if (permitted(items[i].gpu_id)) items[kept++] = items[i];
    // Zero excluded trailing records, but retain every real allowed aperture field.
    memset(items + kept, 0, (count - kept) * sizeof(*items));
    fprintf(stderr, "RLK_KFD_APERTURE_ISOLATION input=%u kept=%u\n", count, kept);
    return kept;
}

int ioctl(int fd, unsigned long request, ...) {
    va_list args;
    va_start(args, request);
    void *arg = va_arg(args, void *);
    va_end(args);
    int selected_query = request == AMDKFD_IOC_GET_PROCESS_APERTURES_NEW ||
                         request == AMDKFD_IOC_GET_PROCESS_APERTURES ||
                         request == AMDKFD_IOC_SET_MEMORY_POLICY ||
                         request == AMDKFD_IOC_ACQUIRE_VM;
    struct stat st;
    if (!selected_query || !kfd_device || fstat(fd, &st) ||
        !S_ISCHR(st.st_mode) || st.st_rdev != kfd_device)
        return next_ioctl(fd, request, arg);
    if (arg && request == AMDKFD_IOC_SET_MEMORY_POLICY &&
        !permitted(((struct kfd_ioctl_set_memory_policy_args *)arg)->gpu_id)) {
        errno = EPERM;
        return -1;
    }
    if (arg && request == AMDKFD_IOC_ACQUIRE_VM &&
        !permitted(((struct kfd_ioctl_acquire_vm_args *)arg)->gpu_id)) {
        errno = EPERM;
        return -1;
    }
    uint32_t capacity = 0;
    if (arg && request == AMDKFD_IOC_GET_PROCESS_APERTURES_NEW)
        capacity = ((struct kfd_ioctl_get_process_apertures_new_args *)arg)->num_of_nodes;
    int result = next_ioctl(fd, request, arg);
    if (result || !arg) return result;
    if (request == AMDKFD_IOC_GET_PROCESS_APERTURES_NEW) {
        struct kfd_ioctl_get_process_apertures_new_args *a = arg;
        if (a->kfd_process_device_apertures_ptr && capacity) {
            if (a->num_of_nodes > capacity) { errno = EOVERFLOW; return -1; }
            a->num_of_nodes = filter((void *)(uintptr_t)a->kfd_process_device_apertures_ptr,
                                    a->num_of_nodes);
        }
    } else if (request == AMDKFD_IOC_GET_PROCESS_APERTURES) {
        struct kfd_ioctl_get_process_apertures_args *a = arg;
        if (a->num_of_nodes > NUM_OF_SUPPORTED_GPUS) { errno = EOVERFLOW; return -1; }
        a->num_of_nodes = filter(a->process_apertures, a->num_of_nodes);
    }
    return result;
}

#ifdef RLK_TEST
uint32_t rlk_test_filter(struct kfd_process_device_apertures *items, uint32_t count) {
    return filter(items, count);
}
#endif
