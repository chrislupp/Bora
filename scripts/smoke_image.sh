#!/usr/bin/env bash
# Smoke-test a locally loaded bora_server image on an ephemeral host port.
#
# Usage: scripts/smoke_image.sh IMAGE   (needs philote-mdo in the active Python)
set -euo pipefail
image="$1"
here="$(cd "$(dirname "$0")" && pwd)"

container=$(docker run --rm -d -p "127.0.0.1::50051" "$image")
cleanup() {
    docker logs "$container" || true
    docker stop "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

address=$(docker port "$container" 50051/tcp | head -n1)
python3 "$here/smoke_server.py" --address "$address"
