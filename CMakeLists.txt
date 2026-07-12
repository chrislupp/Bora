cmake_minimum_required(VERSION 3.20)

# Set CMake policies for package root variables
cmake_policy(SET CMP0074 NEW)  # Use <PackageName>_ROOT variables
cmake_policy(SET CMP0144 NEW)  # Use upper-case <PACKAGENAME>_ROOT variables

# Set macOS deployment target to current system version to avoid version mismatches
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    execute_process(
        COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE MACOS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Extract major.minor version (e.g., "15.0.1" -> "15.0")
    string(REGEX REPLACE "^([0-9]+\\.[0-9]+).*" "\\1" MACOS_VERSION "${MACOS_VERSION}")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${MACOS_VERSION}" CACHE STRING "Minimum macOS deployment version")
    message(STATUS "Setting macOS deployment target to: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

# Project settings
project(Bora
    VERSION 0.1.0
    DESCRIPTION "A port of NeuralFoil"
    LANGUAGES CXX
)

# CMake modules
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Cache option variables
option(BUILD_TESTS "Build unit and integration tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BORA_NATIVE_ARCH "Enable native architecture optimizations (-march=native)" ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Create INTERFACE library for cross-platform compiler options
add_library(project_compiler_options INTERFACE)
add_library(Bora::compiler_options ALIAS project_compiler_options)

# Release optimizations - cross-platform using generator expressions
target_compile_options(project_compiler_options INTERFACE
    # GCC/Clang options
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Release>>:-O3;-fPIC>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Release>,$<BOOL:${BORA_NATIVE_ARCH}>>:-march=native;-fPIC>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Debug>>:-g;-O0;-fPIC>
    # MSVC options
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od;/Zi>
)

target_compile_definitions(project_compiler_options INTERFACE
    $<$<CONFIG:Release>:NDEBUG>
)

# Packages
# find_package(<dependency> REQUIRED)

# Sub-directories
add_subdirectory(include)
add_subdirectory(src)

# Install CMake package configuration files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BoraConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/BoraConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Bora"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/BoraConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/BoraConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/BoraConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Bora"
)

# Install the compiler options interface (empty after install, build-time only)
install(TARGETS project_compiler_options
    EXPORT BoraTargets
)

install(EXPORT BoraTargets
    FILE BoraTargets.cmake
    NAMESPACE Bora::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Bora"
)

# Testing
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
