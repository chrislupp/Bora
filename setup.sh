#!/usr/bin/env bash
#
# setup.sh — Configure this C++ project template.
#
# Replaces placeholder names across all files, renames source files,
# and optionally initializes git with main/develop branches.
#
set -euo pipefail

# ── Prompt for inputs ────────────────────────────────────────────────

read -rp "Project name (PascalCase, e.g. MyProject): " PROJECT_NAME
if [[ -z "$PROJECT_NAME" ]]; then
    echo "Error: project name cannot be empty." >&2
    exit 1
fi

read -rp "Author name: " AUTHOR_NAME
if [[ -z "$AUTHOR_NAME" ]]; then
    echo "Error: author name cannot be empty." >&2
    exit 1
fi

read -rp "Project description: " PROJECT_DESC
read -rp "GitHub username or org: " GITHUB_USER

# ── Derive name variants ────────────────────────────────────────────

# lowercase: MyProject -> myproject
LOWER_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]')

# UPPER_CASE: MyProject -> MYPROJECT
UPPER_NAME=$(echo "$PROJECT_NAME" | tr '[:lower:]' '[:upper:]')

YEAR=$(date +%Y)

echo ""
echo "Configuring template with:"
echo "  Project name:  $PROJECT_NAME"
echo "  Lowercase:     $LOWER_NAME"
echo "  Uppercase:     $UPPER_NAME"
echo "  Author:        $AUTHOR_NAME"
echo "  Description:   $PROJECT_DESC"
echo "  GitHub user:   $GITHUB_USER"
echo ""

# ── Replace placeholders in files ────────────────────────────────────

# Files to process (exclude .git, build, and this script itself)
find_files() {
    find . \
        -not -path './.git/*' \
        -not -path './build/*' \
        -not -path './_deps/*' \
        -not -name 'setup.sh' \
        -type f
}

echo "Replacing placeholders in files..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_INPLACE=(sed -i '')
else
    SED_INPLACE=(sed -i)
fi

while IFS= read -r file; do
    # Skip binary files
    if file "$file" | grep -q "binary"; then
        continue
    fi

    "${SED_INPLACE[@]}" \
        -e "s/ProjectName/${PROJECT_NAME}/g" \
        -e "s/projectname/${LOWER_NAME}/g" \
        -e "s/PROJECTNAME/${UPPER_NAME}/g" \
        -e "s/GITHUB_USER/${GITHUB_USER}/g" \
        -e "s/Project description/${PROJECT_DESC}/g" \
        -e "s/AUTHOR_NAME/${AUTHOR_NAME}/g" \
        "$file" 2>/dev/null || true
done < <(find_files)

# ── Rename files containing 'projectname' ────────────────────────────

echo "Renaming files..."

rename_file() {
    local old="$1"
    local new="${old//projectname/$LOWER_NAME}"
    new="${new//ProjectName/$PROJECT_NAME}"
    if [[ "$old" != "$new" ]]; then
        mv "$old" "$new"
        echo "  $old -> $new"
    fi
}

# Rename source files
for f in \
    "./include/projectname.h" \
    "./src/projectname.cpp" \
    "./test/test_projectname.cpp" \
    "./cmake/ProjectNameConfig.cmake.in"; do
    if [[ -f "$f" ]]; then
        rename_file "$f"
    fi
done

# ── Initialize git (optional) ───────────────────────────────────────

echo ""
read -rp "Initialize git with main/develop branches? [Y/n]: " INIT_GIT
INIT_GIT="${INIT_GIT:-Y}"

if [[ "$INIT_GIT" =~ ^[Yy]$ ]]; then
    # Remove existing git history (this is a template)
    rm -rf .git

    git init -b main
    git add -A
    git commit -m "Initial project setup from CppTemplate"

    git checkout -b develop
    echo "Git initialized with 'main' and 'develop' branches (on develop)."
fi

# ── Clean up ─────────────────────────────────────────────────────────

echo ""
read -rp "Delete this setup script? [Y/n]: " DELETE_SELF
DELETE_SELF="${DELETE_SELF:-Y}"

if [[ "$DELETE_SELF" =~ ^[Yy]$ ]]; then
    rm -- "$0"
    echo "Setup script removed."
fi

echo ""
echo "Done! Your project '$PROJECT_NAME' is ready."
echo ""
echo "Next steps:"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build"
echo "  ctest --test-dir build"
