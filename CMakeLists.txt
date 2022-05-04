cmake_minimum_required(VERSION 3.18)
project(cuMpSGEMM LANGUAGES CXX CUDA)

set(BUILD_SHGEMM_TEST OFF)

find_package(CUDA 11.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 80 86)

# Directories
set(INCDIR include)
set(SRCDIR src)
set(SUBMODULEDIR submodules)

file(GLOB HEADERS "${INCDIR}/cumpsgemm/*.hpp")
file(GLOB INSTANCES "${SRCDIR}/instances/*.cu")

## main library
add_library(cumpsgemm SHARED
	${SRCDIR}/cumpsgemm.cu
	${INSTANCES}
	${HEADERS}
	)

target_include_directories(cumpsgemm PUBLIC ${INCDIR} ${SUBMODULEDIR}/cutf/include ${SUBMODULEDIR}/wmma_extension/include)
target_link_libraries(cumpsgemm PRIVATE
	cuda
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
	cumpsgemm
	cuda
	curand
	)
