# AI Agent Guidelines for ProjectName

## Project Overview

ProjectName is a C++20 library. Replace this description after running setup.sh.

## Build System

### Requirements
- CMake 3.20+
- C++20 compatible compiler
- GTest (for unit tests)

### Build Commands
```bash
# Configure (from repository root)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all
cmake --build build

# Build specific target
cmake --build build --target <target_name>

# Run all tests
ctest --test-dir build

# Run specific test executable
./build/test/ProjectNameTests
```

### Linting and Formatting
Before pushing any changes to the remote, always run formatting and linting checks:

```bash
# Check formatting (must pass before push)
find include src -name '*.h' -o -name '*.cpp' | xargs clang-format --dry-run --Werror

# Auto-fix formatting
find include src -name '*.h' -o -name '*.cpp' | xargs clang-format -i

# Run clang-tidy (requires cmake configure with compile commands)
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
run-clang-tidy -p build -header-filter='include/' 'src/.*\.cpp$'
```

### Key CMake Options
- `BUILD_TESTS=ON|OFF` - Enable/disable test building (default: ON)
- `BUILD_EXAMPLES=ON|OFF` - Enable/disable example programs (default: ON)
- `PROJECTNAME_NATIVE_ARCH=ON|OFF` - Enable native architecture optimizations (default: ON)

## Documentation Site

The `docs/` directory contains a Docusaurus site for project documentation.

### Local Development
```bash
cd docs
npm install
npm start        # Dev server on :3000 with hot reload
npm run build    # Production build to docs/build/
npm run serve    # Serve production build locally
```

### Structure
- `docs/docs/` -- Markdown content (getting-started, guide, api, examples)
- `docs/src/` -- Custom React components and CSS
- `docs/static/` -- Static assets (images, favicon)
- `docs/docusaurus.config.ts` -- Site configuration
- `docs/sidebars.ts` -- Navigation structure

### Versioning
Doc versions are created automatically by the release workflow for stable releases. Manual versioning:
```bash
cd docs
npx docusaurus docs:version X.Y.Z
```

## Code Organization

```
include/               # Public headers
src/                   # Source files
test/                  # Unit tests (GTest)
examples/              # Example programs
scripts/               # Utility scripts
cmake/                 # CMake configuration templates
docs/                  # Docusaurus documentation site
.github/workflows/     # CI/CD workflows
```

## CI/CD

### Build Workflow (`.github/workflows/build.yml`)
Runs on every push and PR to `main`/`develop`:
- Formatting check (clang-format)
- Clang-tidy (PRs to main only)
- Build and test matrix

### Docs Workflow (`.github/workflows/docs.yml`)
Deploys documentation to GitHub Pages when changes are pushed to `develop` in:
- `docs/**` (documentation content or config)
- `include/**` (public headers that may affect API docs)

### Release Workflow (`.github/workflows/release.yml`)
Label-driven release pipeline. Triggered when a labeled PR is merged to `main`.

**Label system** (applied to PRs targeting `main`):

| Purpose | Labels (pick exactly one per row) |
|---------|-----------------------------------|
| Release type | `release` OR `prerelease` |
| Version bump | `major`, `minor`, or `patch` |
| Pre-release qualifier | `alpha`, `beta`, or `rc` (only with `prerelease`) |

**What the workflow does on merge:**
1. Extracts version from `CMakeLists.txt` and bumps per labels
2. Updates copyright years (stable releases only, via `scripts/update_copyright.py`)
3. Updates `VERSION` in `CMakeLists.txt`
4. Updates `CHANGELOG.md` (converts `[Unreleased]` to versioned section)
5. Creates a signed commit and annotated tag
6. Pushes to `main` and creates a GitHub Release

**Requirements**: A PAT with `repo` scope stored as `RELEASE_TOKEN` in Actions secrets.

### Changelog
`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/) format. New changes go under `[Unreleased]`. The release workflow converts this to a versioned section automatically.

**Important**: When adding, changing, or removing features, always update `CHANGELOG.md` as part of the same commit. Add a concise entry under the appropriate subsection of `[Unreleased]`:
- `### Added` — new features or capabilities
- `### Changed` — changes to existing functionality
- `### Deprecated` — features that will be removed in a future release
- `### Removed` — features that were removed
- `### Fixed` — bug fixes
- `### Security` — vulnerability fixes

## Coding Conventions

### Style
- Use 4-space indentation
- Opening braces on same line for control structures
- Member variables use trailing underscore: `member_variable_`
- Use `size_t` for sizes and indices
- Prefer `auto` for complex types, explicit types for primitives

### Header Guards
Use `#pragma once` (no traditional include guards)

### Documentation
- Doxygen-style comments for public APIs
- Brief `@brief` description required
- Document parameters with `@param`
- Document return values with `@return`

## Testing

### Test Framework
Uses Google Test (gtest). Tests are organized by component.

### Running Tests
```bash
# All tests
ctest --test-dir build

# Specific test with filter
./build/test/ProjectNameTests --gtest_filter="*TestName*"
```

## Pull Requests

When creating PRs:
- **Label** with the appropriate type: `enhancement`, `bug`, `documentation`, etc.
- Target `develop` for feature work, `main` only for releases

## Git Workflow

- **`main`**: Stable releases only. PRs from `develop` with release labels.
- **`develop`**: Integration branch. Feature branches merge here.
- **`feature/*`**: Feature branches off `develop`.

## Creating a Release
1. Ensure `CHANGELOG.md` has entries under `[Unreleased]`
2. Create a PR from `develop` → `main`
3. Add labels: one of `release`/`prerelease`, one of `major`/`minor`/`patch`, and optionally a pre-release qualifier
4. Merge the PR — the release workflow handles the rest
