import pyopencl as cl

# Lấy tất cả các nền tảng OpenCL
platforms = cl.get_platforms()

for platform in platforms:
    # Lấy các thiết bị (device) của mỗi nền tảng
    devices = platform.get_devices()

    for device in devices:
        # In thông tin về thiết bị
        print(f"Device Name: {device.name}")
        print(f"Vendor: {device.vendor}")
        print(f"Version: {device.version}")
        print(f"OpenCL C version: {device.opencl_c_version}")
        print(f"Driver Version: {device.driver_version}")
        print(f"Global Memory Size: {device.global_mem_size // (1024 ** 2)} MB")
        print(f"Max Compute Units: {device.max_compute_units}")
        print(f"Max Work Group Size: {device.max_work_group_size}")
        print(f"Max Clock Frequency: {device.max_clock_frequency} MHz")
        print(f"Max Memory Allocation: {device.max_mem_alloc_size // (1024 ** 2)} MB")
        print("=========================================")
