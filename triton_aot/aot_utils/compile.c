// Based on https://github.com/openai/triton/commit/a767ca41e189988740d35cbb9aecd873c4874a62
// THIS IS JUST A TEMPLATE FILE THAT WILL BE USED TO GENERATE THE ACTUAL C CODE

/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <cuda.h>
#include "utils.h"

// helpers to check for cuda errors
#define CUDA_CHECK(ans) {{\
    gpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void gpuAssert(CUresult code, const char *file, int line) {{
  if (code != CUDA_SUCCESS) {{
    const char *prefix = "Triton Error [CUDA]: ";
    const char *str;
    cuGetErrorString(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\n", err);
    exit(code);
  }}
}}

// globals
{cubins}

void unload_{kernel_name}(void) {{
    int device_cc = getDeviceCC();

    switch (device_cc) {{
{unload_switch}
      default:
        printf("Triton Error: Unsupported device cc %d\n", device_cc);
        exit(1);
    }}
}}

// TODO: some code duplication with `runtime/backend/cuda.c`
void load_{kernel_name}() {{
    int dev = 0;
    int device_cc = getDeviceCC();
    void *bin;
    int shared_optin;

    switch (device_cc) {{
{load_switch}
      default:
        printf("Triton Error: Unsupported device cc %d\n", device_cc);
        exit(1);
    }}
}}

/*
{kernel_docstring}
*/
CUresult {kernel_name}(CUstream stream, {signature}) {{
    int device_cc = getDeviceCC();
    CUresult result;

    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    if(gX * gY * gZ > 0) {{
      switch (device_cc) {{
{launch_switch}
        default:
          printf("Triton Error: Unsupported device cc %d\n", device_cc);
          exit(1);
      }}
    }}
    return result;
}}
