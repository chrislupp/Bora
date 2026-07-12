#include "projectname.h"

#include <iostream>

int main() {
    std::cout << "ProjectName version: " << projectname::GetVersion()
              << std::endl;
    std::cout << "2 + 3 = " << projectname::Add(2, 3) << std::endl;
    return 0;
}
