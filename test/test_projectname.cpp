#include "projectname.h"

#include <gtest/gtest.h>

TEST(ProjectNameTest, VersionIsNotEmpty) {
    EXPECT_FALSE(projectname::GetVersion().empty());
}

TEST(ProjectNameTest, AddPositive) {
    EXPECT_EQ(projectname::Add(2, 3), 5);
}

TEST(ProjectNameTest, AddNegative) {
    EXPECT_EQ(projectname::Add(-1, -2), -3);
}

TEST(ProjectNameTest, AddZero) {
    EXPECT_EQ(projectname::Add(0, 0), 0);
}
