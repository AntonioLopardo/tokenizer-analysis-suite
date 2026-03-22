#!/usr/bin/env bash
# Standalone pull for this repo's big files.
# For the full CLI (add/rm/push/status), use the root repo's scripts/bigfiles.sh.
set -euo pipefail

MANIFEST=".bigfiles.json"
[ -f "$MANIFEST" ] || { echo "No $MANIFEST found in $(pwd)"; exit 1; }
command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' CLI not found. Install: pip install huggingface_hub"; exit 1; }

bucket=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['bucket'])")
count=0

python3 -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
for path, info in data.get('files', {}).items():
    print(f\"{path}\t{info['sha256']}\t{info['bucket_path']}\")
" | while IFS=$'\t' read -r path expected_sha bpath; do
  mkdir -p "$(dirname "$path")"
  echo "Pulling: $bucket/$bpath -> $path"
  hf buckets cp "$bucket/$bpath" "$path"

  # Verify hash
  actual_sha=$(shasum -a 256 "$path" 2>/dev/null || sha256sum "$path" | awk '{print $1}')
  actual_sha=$(echo "$actual_sha" | awk '{print $1}')
  if [ "$actual_sha" != "$expected_sha" ]; then
    echo "WARNING: sha256 mismatch for $path"
  else
    echo "Verified: $path"
  fi
  count=$((count + 1))
done

echo "Done."
