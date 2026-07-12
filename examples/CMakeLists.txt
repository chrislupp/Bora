add_executable(example example.cpp)

target_link_libraries(example
    Bora
    Bora::compiler_options
)

add_executable(optimize_airfoil optimize_airfoil.cpp)

target_link_libraries(optimize_airfoil
    Bora
    Bora::compiler_options
)
