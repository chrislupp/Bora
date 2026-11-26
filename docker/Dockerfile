# syntax=docker/dockerfile:1.7

# Bora Philote discipline server (bora_server, gRPC on port 50051).
#
#   docker build -f docker/Dockerfile -t bora .
#   docker run --rm -p 50051:50051 bora
#
# The build needs no system gRPC or protobuf: the philote-mdo crate vendors protoc.
# The image builds natively for linux/amd64 and linux/arm64; on x86-64 the server
# picks its AVX2/FMA kernels at run time, so one image serves every host.

ARG RUST_IMAGE=rust:1.91-slim-trixie
ARG RUNTIME_IMAGE=debian:trixie-slim

# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
FROM ${RUST_IMAGE} AS builder

WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --release --locked --features philote --bin bora_server \
    && install -D target/release/bora_server /out/bin/bora_server

# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE} AS runtime

LABEL org.opencontainers.image.title="Bora" \
      org.opencontainers.image.description="NeuralFoil airfoil analysis with analytic gradients, served as a Philote discipline" \
      org.opencontainers.image.source="https://github.com/chrislupp/Bora" \
      org.opencontainers.image.documentation="https://chrislupp.github.io/Bora/" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN useradd --system --create-home --shell /usr/sbin/nologin app

# The weight directory recorded at compile time does not exist in this stage.
COPY data /usr/local/share/bora/data
ENV BORA_DATA_DIR=/usr/local/share/bora/data
COPY --from=builder /out/bin/bora_server /usr/local/bin/bora_server

USER app
WORKDIR /home/app

EXPOSE 50051
ENTRYPOINT ["/usr/local/bin/bora_server"]
CMD ["0.0.0.0:50051"]
