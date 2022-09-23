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

chc.enable_exp_stats()
chc.set_exp_stats_params(1., 10.)

for compute_mode in compute_mode_list:
    chc.set_compute_mode(compute_mode)
    cupy.matmul(a, b)

    for e in chc.get_last_exp_stats():
        print(e)

chc.disable_exp_stats()
