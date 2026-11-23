# AI Agent Guidelines for Bora

## Project Overview

Bora is a Rust port of NeuralFoil (formerly C++20; ported to Rust in 2026-09).

## Build System

### Requirements
- Rust 1.89+ (edition 2024; install via rustup)
- No system libraries: HDF5 weight files are read with the pure-Rust `hdf5-pure` crate

### Build Commands
```bash
# Build library, binaries, and examples
cargo build --release

# Run all tests (the Philote tests require --all-features)
cargo test --all-features

# Run a specific integration test file / filter by name
cargo test --test neuralfoil
cargo test --all-features -- matches_python

# Run the kernel benchmarks (all model sizes, or only the named ones)
cargo bench
cargo bench -- medium xlarge

# Run examples
cargo run --release --example example
cargo run --release --example polar
cargo run --release --example optimize_airfoil
```

### Linting and Formatting
Before pushing any changes to the remote, always run formatting and linting checks:

```bash
# Check formatting (must pass before push)
cargo fmt --all --check

# Auto-fix formatting
cargo fmt --all

# Run clippy (warnings are errors in CI)
cargo clippy --all-targets --all-features -- -D warnings
```

### Cargo Features
- `philote` - Build the Philote discipline (`bora::philote`) and the `bora_server` gRPC binary (default: off; uses the `philote-mdo` crate, which vendors `protoc`, so no system gRPC/protobuf install is needed)

### Performance
- FMA: aarch64 always has it. On x86-64, `dispatch` in `src/network.rs` runs each network pass through an `#[target_feature(enable = "avx2,fma")]` copy when `is_x86_feature_detected!` reports AVX2 and FMA (decided once per process), else the SSE2 baseline; `RUSTFLAGS="-C target-cpu=native"` compiles AVX2/FMA in statically and removes the check. Kernels take the multiply-add strategy as a type parameter (`M: MulAdd`, `Fused`/`Unfused`), because `cfg(target_feature)` inside a `#[target_feature]` function still reflects the crate-wide flags. Everything a pass calls must stay `#[inline(always)]`: a non-inlined callee is compiled without AVX2/FMA, and `f64::mul_add` there becomes a slow libm call
- Testing the x86-64 baseline path on an AVX2 machine: set `BORA_DISABLE_AVX2_FMA=1` (hidden switch, read once per process; not public API), e.g. `BORA_DISABLE_AVX2_FMA=1 cargo test --all-features`
- The network kernels in `src/network.rs` (the `MulAdd` strategies, the 16-accumulator `dot`/`dot2`, the register-accumulated `jacobian_row` with seed counts padded to a multiple of 4, and the `Lanes` kernel-per-type dispatch) are hand-tuned; small rewrites (e.g. selecting a kernel with `if B == 2`, indexing generic `[[f64; K]; B]` accumulators) measured 15-100% slower
- The reverse-mode kernels `adjoint_rows` (`ADJOINT_ROW_BLOCK` = 4 weight rows per update, up to `MAX_ADJOINT_LANES` = 16 lanes shared across inputs) and `dense_layer_slopes` (values plus swish slopes in one forward traversal) are hand-tuned as well; see the measurements in their doc comments
- The tangent pass (`dense_layer_tangent`: value and tangent lanes through `dot2`) and the batched kernel (`dense_block`: `BATCH_BLOCK_SIZE` points x 4 hand-unrolled output rows, `exp_vectorizable` for swish) are hand-tuned too: sharing one weight traversal between the original and flipped tangent passes, a 4-vector dot, and a generic `[[f64; B]; R]` accumulator block all measured slower (see their doc comments)
- `cargo bench` (`benches/kernels.rs`, std only) times the analysis calls (including the tangent and batched ones) and the network passes for every model size; `cargo bench -- medium xlarge` restricts the sizes. Compare against the previous version before and after any kernel change

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
- `docs/docs/` -- Markdown content (getting-started, guide, background, examples, api, references)
- `docs/src/` -- Custom React components and CSS
  - `src/css/custom.css` -- the theme; its header comment states the design rules
  - `src/components/` -- figures drawn in code (`plot.tsx` holds the shared graph-sheet pieces)
  - `src/data/polar.json` -- figure data; regenerate with
    `cargo run --release --example polar -- --json > docs/src/data/polar.json`
- `docs/static/` -- Static assets (favicon, self-hosted fonts with their OFL licenses)

### Content and design rules
- Facts about NeuralFoil (how it works, training, accuracy) are paraphrased and cited to
  `docs/docs/references.md` at the point of use; never copy upstream text
- Program output shown in the docs is real: rerun the example and paste the result
- Theme: paper, black ink, and the orange ruling of a graph sheet. Orange rules the plates and
  underlines links and selections; it never sets text. Charts are scaled so the sheet's rulings
  are the chart's grid; series differ by symbol and states by line type, not colour. The grid
  stays inside plates, never behind running text
- New doc pages exist only under `/docs/next/` until a release; links from the home page go
  through `src/lib/useDocPath.ts`, and footer links must point at pages that exist in the
  latest released version
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
src/                   # Library crate (lib.rs, types, kulfan, network, neuralfoil, philote)
src/bin/               # bora_server (requires the `philote` feature)
tests/                 # Integration tests; tests/common/ holds shared loaders
tests/data/            # Committed HDF5 reference data from Python NeuralFoil
benches/               # std-only timing harness (`cargo bench`)
examples/              # Example programs
data/                  # Network weights (HDF5)
scripts/               # Weight conversion, reference data generation, release helpers
docs/                  # Docusaurus documentation site
.github/workflows/     # CI/CD workflows
```

## CI/CD

### Build Workflow (`.github/workflows/build.yml`)
Runs on every push and PR to `main`/`develop`:
- Formatting check (`cargo fmt`)
- Clippy (`-D warnings`, all targets and features)
- Build and test matrix (Ubuntu and macOS, stable and MSRV 1.89)

### Docs Workflow (`.github/workflows/docs.yml`)
Deploys documentation to GitHub Pages when changes are pushed to `develop` in:
- `docs/**` (documentation content or config)
- `src/**` (sources whose API may affect the docs)

### Release Workflow (`.github/workflows/release.yml`)
Label-driven release pipeline. Triggered when a labeled PR is merged to `main`.

**Label system** (applied to PRs targeting `main`):

| Purpose | Labels (pick exactly one per row) |
|---------|-----------------------------------|
| Release type | `release` OR `prerelease` |
| Version bump | `major`, `minor`, or `patch` |
| Pre-release qualifier | `alpha`, `beta`, or `rc` (only with `prerelease`) |

**What the workflow does on merge:**
1. Extracts version from `Cargo.toml` and bumps per labels
2. Updates copyright years (stable releases only, via `scripts/update_copyright.py`)
3. Updates `version` in `Cargo.toml` (and the package entry in `Cargo.lock`)
4. Updates `CHANGELOG.md` (converts `[Unreleased]` to versioned section)
5. Creates a signed commit and annotated tag
6. Pushes to `main` and creates a GitHub Release

### Publish Workflow (`.github/workflows/publish.yml`)
Publishes to crates.io when a stable GitHub Release is published (pre-releases are skipped,
since their Cargo.toml version has no qualifier); rerun manually with `workflow_dispatch` and a
tag. Uses crates.io trusted publishing (OIDC, `crates-io` environment); a `CARGO_REGISTRY_TOKEN` secret, if present, is used instead (bootstrap for the first publish only; 0.1.0 is not on crates.io). The
package is `bora-foil` on crates.io (`bora` is taken) with `[lib] name = "bora"`; keep the
`.crate` under the 10 MB upload cap (`cargo package --list`, `exclude` in `Cargo.toml`).

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
- Default `rustfmt` formatting (4-space indentation, 100-column lines)
- Standard Rust naming: `snake_case` functions and fields, `CamelCase` types, `SCREAMING_SNAKE_CASE` constants
- Constructors and loaders return `bora::Result<T>`; analysis calls are infallible
- Upstream NeuralFoil names (`CL`, `Top_Xtr`, `TE_thickness`, ...) appear only as Philote variable name strings

### Documentation
- `///` rustdoc comments on all public items
- Document failure modes under `# Errors` and panics under `# Panics`

## Testing

### Test Framework
Cargo integration tests in `tests/`, organized by component. Reference data lives in committed
HDF5 files under `tests/data/` (regenerate with `scripts/generate_*_ref_data.py` only when
updating the reference data). `[profile.test]` uses `opt-level = 3` so the reference suites run
quickly.

### Running Tests
```bash
# All tests
cargo test --all-features

# Specific test with filter
cargo test --all-features -- TestName
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
