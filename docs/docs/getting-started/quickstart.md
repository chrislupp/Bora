---
sidebar_position: 2
title: "Quickstart"
---

# Quickstart

This guide walks through a minimal ProjectName program.

## Complete Example

```cpp
#include <projectname.h>
#include <iostream>

int main() {
    // TODO: Replace with a real usage example of your library

    std::cout << "Hello from ProjectName!" << std::endl;

    return 0;
}
```

## Step-by-Step Explanation

### 1. Include Headers

Include the main ProjectName header to access the library's API:

```cpp
#include <projectname.h>
```

### 2. Use the Library

<!-- TODO: Add step-by-step explanations of the example code -->

Replace this section with a walkthrough of your library's core functionality.

### 3. Build and Run

Add ProjectName to your CMakeLists.txt:

```cmake
find_package(ProjectName REQUIRED)
target_link_libraries(my_app PRIVATE projectname)
```

Or if building within the ProjectName project tree:

```cmake
target_link_libraries(my_app PRIVATE projectname)
```
