import openvino as ov

core = ov.Core()
available_devices = core.available_devices
print(available_devices)