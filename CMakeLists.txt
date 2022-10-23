cmake_minimum_required(VERSION 3.18)
project(cuMpSGEMM LANGUAGES CXX CUDA)

find_package(CUDA 11.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 80)

# Directories
set(INCDIR include)
set(SRCDIR src)
set(SUBMODULEDIR submodules)

file(GLOB HEADERS "${INCDIR}/cumpsgemm/*.hpp")

set(
	LIBSRCS
	${SRCDIR}/exp_stats.cu
	${SRCDIR}/cumpsgemm.cu
	${SRCDIR}/cumpsgemm_cublas.cu
	${SRCDIR}/default_cumpsgemm_rule.cu
	${SRCDIR}/handle.cu
	${SRCDIR}/dynamic_launch.cu
	${SRCDIR}/dynamic_scaling.cu
	${SRCDIR}/culip.cu
	${SUBMODULEDIR}/cuGEMM-Mx2x2/src/main.cu
	${HEADERS}
	)

add_library(libobjs OBJECT ${LIBSRCS})
set_property(TARGET libobjs PROPERTY POSITION_INDEPENDENT_CODE 1)
target_include_directories(libobjs PUBLIC ${INCDIR} ${SUBMODULEDIR}/cutf/include ${SUBMODULEDIR}/wmma_extension/include ${SUBMODULEDIR}/cuGEMM-Mx2x2/include)

## static library
add_library(cumpsgemm_static STATIC $<TARGET_OBJECTS:libobjs>)
target_link_libraries(cumpsgemm_static PRIVATE
	cublas_static
	cublasLt_static
	culibos
	)

#target_include_directories(cumpsgemm_static PUBLIC ${INCDIR} ${SUBMODULEDIR}/cutf/include ${SUBMODULEDIR}/wmma_extension/include)

## shared library
add_library(cumpsgemm SHARED $<TARGET_OBJECTS:libobjs>)

#target_include_directories(cumpsgemm PUBLIC ${INCDIR} ${SUBMODULEDIR}/cutf/include ${SUBMODULEDIR}/wmma_extension/include)
target_link_libraries(cumpsgemm PRIVATE
	cublas_static
	cublasLt_static
	culibos
	)


## Default rule library
add_library(cumpsgemm_rule SHARED
	${SRCDIR}/default_cumpsgemm_rule.cu
	${HEADERS}
	)

target_include_directories(cumpsgemm_rule PUBLIC ${INCDIR})
target_link_libraries(cumpsgemm_rule PRIVATE
	cuda
	)

##########################################################################
# Tests
##########################################################################
set(TESTSRCDIR test)
add_executable(cumpsgemm_test ${TESTSRCDIR}/main.cu ${HEADERS})
target_include_directories(cumpsgemm_test PRIVATE ${INCDIR} ${SUBMODULEDIR}/cutf/include ${SUBMODULEDIR}/wmma_extension/include)
target_link_libraries(cumpsgemm_test PRIVATE
	CUDA::cublas
	cumpsgemm_static
	cuda
	curand
	)

add_executable(cumpsgemm_hijack_test ${TESTSRCDIR}/hijack.cu ${HEADERS})
target_include_directories(cumpsgemm_hijack_test PRIVATE ${INCDIR} ${SUBMODULEDIR}/cutf/include)
target_link_libraries(cumpsgemm_hijack_test PRIVATE
	CUDA::cublas
	cumpsgemm_static
	cuda
	curand
	)
