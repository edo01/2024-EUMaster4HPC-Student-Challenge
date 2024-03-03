# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /mnt/tier2/apps/USE/easybuild/release/2023.1/software/CMake/3.26.3-GCCcore-12.3.0/bin/cmake

# The command to remove a file.
RM = /mnt/tier2/apps/USE/easybuild/release/2023.1/software/CMake/3.26.3-GCCcore-12.3.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_CPU_MPI_OMP.out.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_CPU_MPI_OMP.out.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_CPU_MPI_OMP.out.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_CPU_MPI_OMP.out.dir/flags.make

test/CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o: test/CMakeFiles/test_CPU_MPI_OMP.out.dir/flags.make
test/CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o: /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/test/test_CG_CPU_MPI_OMP.cpp
test/CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o: test/CMakeFiles/test_CPU_MPI_OMP.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o"
	cd /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test && /apps/USE/easybuild/release/2023.1/software/GCCcore/12.3.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o -MF CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o.d -o CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o -c /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/test/test_CG_CPU_MPI_OMP.cpp

test/CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.i"
	cd /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test && /apps/USE/easybuild/release/2023.1/software/GCCcore/12.3.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/test/test_CG_CPU_MPI_OMP.cpp > CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.i

test/CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.s"
	cd /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test && /apps/USE/easybuild/release/2023.1/software/GCCcore/12.3.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/test/test_CG_CPU_MPI_OMP.cpp -o CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.s

# Object files for target test_CPU_MPI_OMP.out
test_CPU_MPI_OMP_out_OBJECTS = \
"CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o"

# External object files for target test_CPU_MPI_OMP.out
test_CPU_MPI_OMP_out_EXTERNAL_OBJECTS =

test/test_CPU_MPI_OMP.out: test/CMakeFiles/test_CPU_MPI_OMP.out.dir/test_CG_CPU_MPI_OMP.cpp.o
test/test_CPU_MPI_OMP.out: test/CMakeFiles/test_CPU_MPI_OMP.out.dir/build.make
test/test_CPU_MPI_OMP.out: LAM/libLAM.so
test/test_CPU_MPI_OMP.out: /mnt/tier2/apps/USE/easybuild/release/2023.1/software/impi/2021.9.0-intel-compilers-2023.1.0/mpi/2021.9.0/lib/libmpicxx.so
test/test_CPU_MPI_OMP.out: /mnt/tier2/apps/USE/easybuild/release/2023.1/software/impi/2021.9.0-intel-compilers-2023.1.0/mpi/2021.9.0/lib/release/libmpi.so
test/test_CPU_MPI_OMP.out: /lib64/librt.so
test/test_CPU_MPI_OMP.out: /lib64/libdl.so
test/test_CPU_MPI_OMP.out: /mnt/tier2/apps/USE/easybuild/release/2023.1/software/GCCcore/12.3.0/lib64/libgomp.so
test/test_CPU_MPI_OMP.out: /lib64/libpthread.so
test/test_CPU_MPI_OMP.out: test/CMakeFiles/test_CPU_MPI_OMP.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_CPU_MPI_OMP.out"
	cd /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_CPU_MPI_OMP.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_CPU_MPI_OMP.out.dir/build: test/test_CPU_MPI_OMP.out
.PHONY : test/CMakeFiles/test_CPU_MPI_OMP.out.dir/build

test/CMakeFiles/test_CPU_MPI_OMP.out.dir/clean:
	cd /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_CPU_MPI_OMP.out.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_CPU_MPI_OMP.out.dir/clean

test/CMakeFiles/test_CPU_MPI_OMP.out.dir/depend:
	cd /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/test /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test /home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test/CMakeFiles/test_CPU_MPI_OMP.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_CPU_MPI_OMP.out.dir/depend

