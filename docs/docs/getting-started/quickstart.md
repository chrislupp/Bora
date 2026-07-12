---
sidebar_position: 2
title: "Quickstart"
---

# Quickstart

This guide walks through a minimal Bora program.

## Complete Example

```cpp
#include <bora.h>
#include <iostream>

int main() {
    // TODO: Replace with a real usage example of your library

    std::cout << "Hello from Bora!" << std::endl;

    return 0;
}
```

## Step-by-Step Explanation

### 1. Include Headers

Include the main Bora header to access the library's API:

```cpp
#include <bora.h>
```

### 2. Use the Library

<!-- TODO: Add step-by-step explanations of the example code -->

Replace this section with a walkthrough of your library's core functionality.

### 3. Build and Run

Add Bora to your CMakeLists.txt:

```cmake
find_package(Bora REQUIRED)
target_link_libraries(my_app PRIVATE bora)
```

Or if building within the Bora project tree:

```cmake
target_link_libraries(my_app PRIVATE bora)
```
