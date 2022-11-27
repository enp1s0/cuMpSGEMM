import cumpsgemm_hijack_control as chc
import cupy

a = cupy.random.rand(1000, 1000).astype('f')
b = cupy.random.rand(1000, 1000).astype('f')

compute_mode_list = [
        chc.CUMPSGEMM_CUBLAS,
        chc.CUMPSGEMM_FP16TCEC,
        chc.CUMPSGEMM_TF32TCEC,
        chc.CUMPSGEMM_FP16TC,
        chc.CUMPSGEMM_TF32TC,
        chc.CUMPSGEMM_AUTO,
        ]

compute_mode_name_table = {
        chc.CUMPSGEMM_FP16TCEC : 'CUMPSGEMM_FP16TCEC',
        chc.CUMPSGEMM_TF32TCEC : 'CUMPSGEMM_TF32TCEC',
        chc.CUMPSGEMM_FP16TC   : 'CUMPSGEMM_FP16TC',
        chc.CUMPSGEMM_TF32TC   : 'CUMPSGEMM_TF32TC',
        chc.CUMPSGEMM_CUBLAS   : 'CUMPSGEMM_CUBLAS',
        }

for compute_mode in compute_mode_list:
    print(compute_mode_name_table[compute_mode])
    chc.set_control_function(lambda op_A, op_B, m, n, k : compute_mode)
    cupy.matmul(a, b)
