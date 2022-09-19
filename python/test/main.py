import cumpsgemm_hijack_control as chc
import cupy

a = cupy.random.rand(1000, 1000).astype('f')
b = cupy.random.rand(1000, 1000).astype('f')

compute_mode_list = [
        chc.CUMPSGEMM_FP16TCEC,
        chc.CUMPSGEMM_TF32TCEC,
        chc.CUMPSGEMM_FP16TC,
        chc.CUMPSGEMM_TF32TC,
        ]

for compute_mode in compute_mode_list:
    chc.set_compute_mode(compute_mode)
    cupy.matmul(a, b)
