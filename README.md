# Bora

[![Build and Tests](https://github.com/clupp/Bora/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/clupp/Bora/actions/workflows/build.yml)

A port of NeuralFoil goes here.

## Quick Start

Clone this template and run the setup script:

```bash
./setup.sh
```

The script will prompt for your project name, author, and description, then
configure all files automatically.

## Requirements

- CMake 3.20+
- C++20 compatible compiler (e.g., GCC 12+, Clang 16+)
- GTest (for unit tests)

## Building

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Run tests
ctest --test-dir build
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | `ON` | Enable/disable test building |
| `BUILD_EXAMPLES` | `ON` | Enable/disable example programs |
| `BORA_NATIVE_ARCH` | `ON` | Enable native architecture optimizations |
