add_executable(example example.cpp)

target_link_libraries(example
    Bora
    Bora::compiler_options
)
