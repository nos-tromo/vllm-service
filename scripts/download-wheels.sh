#!/usr/bin/env bash
# Download Python wheels for offline/airgapped Docker builds.
#
# Run this script on an internet-connected machine before creating the
# deployment bundle.  It launches the vllm base image, installs the target
# packages into a throwaway container, and downloads only the packages that
# were not already present in the base image into the project's wheels/ dir.
#
# Usage:
#   ./scripts/download-wheels.sh
#
# After completion, commit the wheel files are git-ignored; transfer the
# wheels/ directory alongside the git bundle to the airgapped server.

set -euo pipefail

VLLM_IMAGE="docker.io/vllm/vllm-openai:v0.17.1@sha256:0dc46f74eb0e630675d83101dc66c6441c4475cceedcf9235ee42b87c3affd23"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="$(dirname "$SCRIPT_DIR")/wheels"

mkdir -p "$WHEELS_DIR"

echo "Pulling base image (skipped if already present locally)..."
docker pull "$VLLM_IMAGE"

echo "Resolving installation delta and downloading wheels to: $WHEELS_DIR"
docker run --rm -i \
    --entrypoint /bin/bash \
    -v "$WHEELS_DIR":/wheels \
    "$VLLM_IMAGE" \
    -s << 'EOF'
set -e

echo "Installing target packages (resolves delta against base image)..."
pip install --quiet \
    --report /tmp/install-report.json \
    'vllm[audio]' orjson 'transformers>=5.5.0' conch-triton-kernels

python3 - << 'PYEOF'
import json, subprocess, sys

with open('/tmp/install-report.json') as f:
    report = json.load(f)

pkgs = [
    f"{p['metadata']['name']}=={p['metadata']['version']}"
    for p in report.get('install', [])
]

if not pkgs:
    print('All packages already satisfied in base image — no wheels to download.')
    sys.exit(0)

print(f'Downloading {len(pkgs)} package(s): {", ".join(pkgs)}')
subprocess.run(
    ['pip', 'download', '--no-deps', '-d', '/wheels'] + pkgs,
    check=True,
)
PYEOF
EOF

COUNT=$(find "$WHEELS_DIR" -name '*.whl' | wc -l)
echo "Done: $COUNT wheel(s) in ./wheels/"
echo
echo "Next steps:"
echo "  1.  tar czf wheels.tar.gz wheels/"
echo "  2.  Transfer wheels.tar.gz alongside the git bundle to the airgapped server."
echo "  3.  On the airgapped server: tar xzf wheels.tar.gz"
echo "  4.  OFFLINE_BUILD=1 docker compose up --build --pull never"
