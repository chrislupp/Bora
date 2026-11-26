#!/usr/bin/env bash
# Publish smoke-tested images, unchanged, as one multi-platform image.
#
# Usage: scripts/publish_image.sh REPOSITORY ARCHIVE_DIR TAG...
#
# ARCHIVE_DIR holds one `docker save` archive per platform (bora-<arch>.tar),
# written by docker-image.yml after the image passed scripts/smoke_image.sh.
# Each archive is pushed by digest (no tag), then one index referencing all of
# them is pushed under the first tag and the remaining tags point at it. The
# pushed images are the archived ones byte for byte; nothing is rebuilt.
# Needs crane, logged in to the registry (docker/login-action is enough).
set -euo pipefail
repository="$1"
archives="$2"
shift 2
(( $# > 0 )) || { echo "no tags given" >&2; exit 1; }

manifests=()
for archive in "$archives"/*.tar; do
    digest=$(crane digest --tarball "$archive")
    crane push "$archive" "$repository@$digest"
    manifests+=(-m "$repository@$digest")
done
(( ${#manifests[@]} > 0 )) || { echo "no image archives in $archives" >&2; exit 1; }

first="$1"
shift
crane index append "${manifests[@]}" -t "$repository:$first"
for tag in "$@"; do
    crane tag "$repository:$first" "$tag"
done
crane manifest "$repository:$first"
echo
echo "Published $repository: $first $*"
