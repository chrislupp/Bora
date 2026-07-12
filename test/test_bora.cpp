#include "bora.h"

#include <gtest/gtest.h>

TEST(BoraTest, VersionIsNotEmpty) {
    EXPECT_FALSE(bora::GetVersion().empty());
}
