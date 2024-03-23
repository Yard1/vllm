#include <cuda.h>

int CC = 0;

int getDeviceCC(){
    if (CC)
      return CC;
    int major;
    int minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
    CC = major * 10 + minor;
    return CC;
}