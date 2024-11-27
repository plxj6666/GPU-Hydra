# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/project/cuda_c++/GPU-Hydra

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/project/cuda_c++/GPU-Hydra/build

# Include any dependencies generated for this target.
include CMakeFiles/test_matrix_isinvertible.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_matrix_isinvertible.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_matrix_isinvertible.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_matrix_isinvertible.dir/flags.make

CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o: CMakeFiles/test_matrix_isinvertible.dir/flags.make
CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o: ../test/test_matrix_isinvertible.cu
CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o: CMakeFiles/test_matrix_isinvertible.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/project/cuda_c++/GPU-Hydra/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o"
	/usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o -MF CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o.d -x cu -dc /root/project/cuda_c++/GPU-Hydra/test/test_matrix_isinvertible.cu -o CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o

CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_matrix_isinvertible
test_matrix_isinvertible_OBJECTS = \
"CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o"

# External object files for target test_matrix_isinvertible
test_matrix_isinvertible_EXTERNAL_OBJECTS =

CMakeFiles/test_matrix_isinvertible.dir/cmake_device_link.o: CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o
CMakeFiles/test_matrix_isinvertible.dir/cmake_device_link.o: CMakeFiles/test_matrix_isinvertible.dir/build.make
CMakeFiles/test_matrix_isinvertible.dir/cmake_device_link.o: libgpu_hydra.a
CMakeFiles/test_matrix_isinvertible.dir/cmake_device_link.o: CMakeFiles/test_matrix_isinvertible.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/project/cuda_c++/GPU-Hydra/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/test_matrix_isinvertible.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_matrix_isinvertible.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_matrix_isinvertible.dir/build: CMakeFiles/test_matrix_isinvertible.dir/cmake_device_link.o
.PHONY : CMakeFiles/test_matrix_isinvertible.dir/build

# Object files for target test_matrix_isinvertible
test_matrix_isinvertible_OBJECTS = \
"CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o"

# External object files for target test_matrix_isinvertible
test_matrix_isinvertible_EXTERNAL_OBJECTS =

test_matrix_isinvertible: CMakeFiles/test_matrix_isinvertible.dir/test/test_matrix_isinvertible.cu.o
test_matrix_isinvertible: CMakeFiles/test_matrix_isinvertible.dir/build.make
test_matrix_isinvertible: libgpu_hydra.a
test_matrix_isinvertible: CMakeFiles/test_matrix_isinvertible.dir/cmake_device_link.o
test_matrix_isinvertible: CMakeFiles/test_matrix_isinvertible.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/project/cuda_c++/GPU-Hydra/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable test_matrix_isinvertible"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_matrix_isinvertible.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_matrix_isinvertible.dir/build: test_matrix_isinvertible
.PHONY : CMakeFiles/test_matrix_isinvertible.dir/build

CMakeFiles/test_matrix_isinvertible.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_matrix_isinvertible.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_matrix_isinvertible.dir/clean

CMakeFiles/test_matrix_isinvertible.dir/depend:
	cd /root/project/cuda_c++/GPU-Hydra/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/project/cuda_c++/GPU-Hydra /root/project/cuda_c++/GPU-Hydra /root/project/cuda_c++/GPU-Hydra/build /root/project/cuda_c++/GPU-Hydra/build /root/project/cuda_c++/GPU-Hydra/build/CMakeFiles/test_matrix_isinvertible.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_matrix_isinvertible.dir/depend

