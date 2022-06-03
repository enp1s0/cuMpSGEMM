#!/bin/sh

TEMPLATE=$(cat <<-EOF
namespace{using DATA_T = %s;}
template void cumpsgemm::launch_kernel<DATA_T, cumpsgemm::SMEM_M,cumpsgemm::SMEM_N,cumpsgemm::SMEM_K, cumpsgemm::FRAG_M,cumpsgemm::FRAG_N,cumpsgemm::FRAG_K, cumpsgemm::BLOCK_SIZE, cumpsgemm::%s,cumpsgemm::%s, %s, mtk::wmma::tcec::%s> (
			const std::size_t m,
			const std::size_t n,
			const std::size_t k,
			const DATA_T alpha,
			const DATA_T* const a_ptr, const std::size_t lda,
			const DATA_T* const b_ptr, const std::size_t ldb,
			const DATA_T beta,
			DATA_T* const c_ptr, const std::size_t ldc,
			cudaStream_t cuda_stream
		);
template void cumpsgemm::launch_stridedBatch_kernel<DATA_T, cumpsgemm::SMEM_M,cumpsgemm::SMEM_N,cumpsgemm::SMEM_K, cumpsgemm::FRAG_M,cumpsgemm::FRAG_N,cumpsgemm::FRAG_K, cumpsgemm::BLOCK_SIZE, cumpsgemm::%s,cumpsgemm::%s, %s, mtk::wmma::tcec::%s> (
			const std::size_t m,
			const std::size_t n,
			const std::size_t k,
			const DATA_T alpha,
			const DATA_T* const a_ptr, const std::size_t lda, const std::size_t stridea,
			const DATA_T* const b_ptr, const std::size_t ldb, const std::size_t strideb,
			const DATA_T beta,
			DATA_T* const c_ptr, const std::size_t ldc, const std::size_t stridec,
			const std::size_t num_blocks_per_gemm,
			cudaStream_t cuda_stream
		);
EOF
)

for T in float cuComplex;do
	for OPA in col_major row_major conjugate;do
		for OPB in col_major row_major conjugate;do
			for TCT in half nvcuda::wmma::precision::tf32;do
				for EC in with_ec without_ec;do
					TCT_name=$(echo ${TCT} | sed -e 's/:://g')
					file_name="ins_${T}_${OPA}_${OPB}_${TCT_name}_${EC}.cu"
					echo "#include \"../cumpsgemm_kernel.cuh\"" > $file_name
					echo $(printf "$TEMPLATE" $T $OPA $OPB $TCT $EC $OPA $OPB $TCT $EC) >> $file_name
				done
			done
		done
	done
done
