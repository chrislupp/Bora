#include "bora.h"

#include <gtest/gtest.h>

TEST(BoraTest, VersionIsNotEmpty) {
    EXPECT_FALSE(bora::GetVersion().empty());
}

TEST(BoraTest, AddPositive) {
    EXPECT_EQ(bora::Add(2, 3), 5);
}

TEST(BoraTest, AddNegative) {
    EXPECT_EQ(bora::Add(-1, -2), -3);
}

TEST(BoraTest, AddZero) {
    EXPECT_EQ(bora::Add(0, 0), 0);
}
