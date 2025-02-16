import numpy as np
import cupy as cp

TEST_COUNT = 100

for matrix_type in ['s', 'c']:
    for N in [2**log_N for log_N in range(5, 12)]:
        residual_sum = 0.
        for t in range(TEST_COUNT):
            mat_A = np.random.random((N, N)).astype('f')
            mat_B = np.random.random((N, N)).astype('f')

            if matrix_type == 'c':
                mat_A = mat_A.astype('complex64') + np.random.random((N, N)).astype('complex64') * 1j
                mat_B = mat_B.astype('complex64') + np.random.random((N, N)).astype('complex64') * 1j

            mat_C_ref = np.dot(mat_A, mat_B)

            dev_mat_A = cp.array(mat_A)
            dev_mat_B = cp.array(mat_B)

            dev_mat_C = cp.dot(dev_mat_A, dev_mat_B)

            dev_mat_C_ref = cp.array(mat_C_ref)

            residual_sum += cp.linalg.norm(dev_mat_C - dev_mat_C_ref) / cp.linalg.norm(dev_mat_C_ref)
        residual = residual_sum / TEST_COUNT
        print('type =', matrix_type, ', N =', N, ', residual =', residual, ('OK' if residual < np.sqrt(N) * 1e-7 else 'NG') + ' as FP32, ', ('OK' if residual < np.sqrt(N) * 1e-4 else 'NG') + ' as FP16')
