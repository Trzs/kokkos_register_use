cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0028 NEW)

# set the project name
project(kokkos_register_use)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
)
FetchContent_MakeAvailable(kokkos)

# add_subdirectory(kokkos)

# add the executable
add_executable(kokkos_register_use main.cpp)

target_link_libraries(kokkos_register_use Kokkos::kokkos)
target_compile_options(kokkos_register_use PRIVATE -Wall -Wextra -Wpedantic)