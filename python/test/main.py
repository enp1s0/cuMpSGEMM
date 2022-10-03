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
        ]

compute_mode_name_table = {
        chc.CUMPSGEMM_FP16TCEC : 'CUMPSGEMM_FP16TCEC',
        chc.CUMPSGEMM_TF32TCEC : 'CUMPSGEMM_TF32TCEC',
        chc.CUMPSGEMM_FP16TC   : 'CUMPSGEMM_FP16TC',
        chc.CUMPSGEMM_TF32TC   : 'CUMPSGEMM_TF32TC',
        chc.CUMPSGEMM_CUBLAS   : 'CUMPSGEMM_CUBLAS',
        }

chc.enable_exp_stats()
chc.set_exp_stats_params(1., 250.)

for compute_mode in compute_mode_list:
    print(compute_mode_name_table[compute_mode])
    chc.set_compute_mode(compute_mode)
    cupy.matmul(a, b)
    buffer_id = chc.get_current_buffer_id()

    print("lost_rate =", chc.get_lost_ratio(buffer_id), ", buffer_id =", buffer_id, ", threshold =", chc.get_global_lost_ratio_threshold())

chc.disable_exp_stats()
