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
include CMakeFiles/test_finite_field.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_finite_field.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_finite_field.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_finite_field.dir/flags.make

CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o: CMakeFiles/test_finite_field.dir/flags.make
CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o: ../test/test_finite_field.cu
CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o: CMakeFiles/test_finite_field.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/project/cuda_c++/GPU-Hydra/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o"
	/usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o -MF CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o.d -x cu -dc /root/project/cuda_c++/GPU-Hydra/test/test_finite_field.cu -o CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o

CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_finite_field
test_finite_field_OBJECTS = \
"CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o"

# External object files for target test_finite_field
test_finite_field_EXTERNAL_OBJECTS =

CMakeFiles/test_finite_field.dir/cmake_device_link.o: CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o
CMakeFiles/test_finite_field.dir/cmake_device_link.o: CMakeFiles/test_finite_field.dir/build.make
CMakeFiles/test_finite_field.dir/cmake_device_link.o: libgpu_hydra.a
CMakeFiles/test_finite_field.dir/cmake_device_link.o: CMakeFiles/test_finite_field.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/project/cuda_c++/GPU-Hydra/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/test_finite_field.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_finite_field.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_finite_field.dir/build: CMakeFiles/test_finite_field.dir/cmake_device_link.o
.PHONY : CMakeFiles/test_finite_field.dir/build

# Object files for target test_finite_field
test_finite_field_OBJECTS = \
"CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o"

# External object files for target test_finite_field
test_finite_field_EXTERNAL_OBJECTS =

test_finite_field: CMakeFiles/test_finite_field.dir/test/test_finite_field.cu.o
test_finite_field: CMakeFiles/test_finite_field.dir/build.make
test_finite_field: libgpu_hydra.a
test_finite_field: CMakeFiles/test_finite_field.dir/cmake_device_link.o
test_finite_field: CMakeFiles/test_finite_field.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/project/cuda_c++/GPU-Hydra/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable test_finite_field"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_finite_field.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_finite_field.dir/build: test_finite_field
.PHONY : CMakeFiles/test_finite_field.dir/build

CMakeFiles/test_finite_field.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_finite_field.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_finite_field.dir/clean

CMakeFiles/test_finite_field.dir/depend:
	cd /root/project/cuda_c++/GPU-Hydra/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/project/cuda_c++/GPU-Hydra /root/project/cuda_c++/GPU-Hydra /root/project/cuda_c++/GPU-Hydra/build /root/project/cuda_c++/GPU-Hydra/build /root/project/cuda_c++/GPU-Hydra/build/CMakeFiles/test_finite_field.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_finite_field.dir/depend

