#!/usr/bin/env bash
set -euo pipefail

# Simple SPIR-V validation script for CI and local use.
# Usage: tools/ci/validate_spirv.sh
# The script composes selected WGSL shaders to SPIR-V using tools/wgsl_to_spv
# and validates them with spirv-val.

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Build the wgsl_to_spv tool if not already built
echo "Building tools/wgsl_to_spv..."
cargo build --manifest-path tools/wgsl_to_spv/Cargo.toml --quiet
WGSL_TO_SPV=target/debug/wgsl_to_spv

if ! command -v spirv-val >/dev/null 2>&1; then
  echo "ERROR: spirv-val not found on PATH. Please install SPIRV-Tools (spirv-val)."
  exit 1
fi

# List of representative WGSL inputs to validate. Add more as needed.
SHADERS=(
  "crates/bevy_pbr/src/render/mesh_preprocess.wgsl"
  "crates/bevy_pbr/src/prepass/prepass_utils.wgsl"
)

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

FAILED=0
for s in "${SHADERS[@]}"; do
  if [[ ! -f "$s" ]]; then
    echo "Warning: shader $s not found, skipping"
    continue
  fi

  OUT="$TMPDIR/$(basename "$s").spv"
  echo "Composing $s -> $OUT"
  "$WGSL_TO_SPV" "$s" "$OUT" || { echo "wgsl_to_spv failed for $s"; FAILED=1; continue; }

  echo "Running spirv-val on $OUT"
  spirv-val --relax-block-layout --target-env vulkan1.3 "$OUT" || { echo "spirv-val failed for $s"; FAILED=1; }
done

if [[ $FAILED -ne 0 ]]; then
  echo "SPIR-V validation failed"
  exit 1
fi

echo "All SPIR-V validated successfully"