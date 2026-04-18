import cupy as cp

print(cp.cuda.runtime.getDeviceCount())   # 设备数

props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"]
if isinstance(name, bytes):
    name = name.decode()

print(name)                               # GPU 名称
print("compute capability:", cp.cuda.Device(0).compute_capability)
print("mem info:", cp.cuda.Device(0).mem_info)