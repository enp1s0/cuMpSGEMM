#!/bin/sh

if [ $# -ne 1 ]; then
	echo "Usage $0 path/to/build"
	exit 1
fi

build_dir=$1

original_cublas_lib=$(dirname $(which nvcc))/../lib64/libcublas_static.a
cumpsgemm_lib=${build_dir}/libcumpsgemm_static.a

hijack_lib_dir=hijack/lib
if [ ! -e "$hijack_lib_dir" ];then
	mkdir -p $hijack_lib_dir
fi

# Create libcublas_static.a
mri=$(cat <<-EOF
create ${hijack_lib_dir}/libcublas_static.a
addlib ${cumpsgemm_lib}
addlib ${original_cublas_lib}
delete sgemm.o
delete cgemm.o
delete gemmEx.o
save
end
EOF
)

echo "Creating a static library for hijacking ..."
echo "--- mri script ---"
echo "$mri"
echo "---            ---"


echo "$mri" | ar -M
if [ "$?" -ne "0" ];then
	echo "Failed to create the static library"
	exit 1
fi

# Copy libcublas.so
echo "Copying a shared library..."
cp ${build_dir}/libcumpsgemm.so ${hijack_lib_dir}

if [ "$?" -ne "0" ];then
	echo "Failed to copy the dynamic library"
	exit 1
fi

echo "Done!"

echo ""
echo "#-- Hijack static library"
echo "export LIBRARY_PATH=$(pwd)/${hijack_lib_dir}:\$LIBRARY_PATH"
echo "// build (e.g. make)"

echo ""
echo "#-- Hijack dynamic library"
echo "// After compiling the target application, "
echo "export LD_PRELOAD=$(pwd)/${hijack_lib_dir}/libcumpsgemm.so:\$LD_PRELOAD"
echo "// and execute it as usual (e.g. ./a.out)"
