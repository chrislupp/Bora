#include "bora.h"

#include <iostream>

int main() {
    std::cout << "Bora version: " << bora::GetVersion()
              << std::endl;
    std::cout << "2 + 3 = " << bora::Add(2, 3) << std::endl;
    return 0;
}
