import ctypes
lib = ctypes.CDLL("./D-ops/main.dll")
lib.init_opencl()
lib.init_snn()

lib.snn_forward.argtypes  = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

test_input1 = (ctypes.c_float * 40)(*range(40))
test_input2 = (ctypes.c_float * 24)(*range(24))
test_input3 = (ctypes.c_float * 6)(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
test_output = (ctypes.c_float * 60)()                    
lib.snn_forward(test_input1, test_input2, test_input3, test_output, 10, 6, 4)
print(list(test_input1))
print(list(test_input2))
print(list(test_output))