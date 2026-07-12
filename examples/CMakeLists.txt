add_executable(example example.cpp)

target_link_libraries(example
    ProjectName
    ProjectName::compiler_options
)
