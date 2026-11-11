#!/bin/sh
# Build the two pinned tau2 environments this project compares, from tau2-bench's own lockfile:
#   .tau2/pre-v1.0.1  (commit b51a6d6, the grader before the July 2026 banking fixes)
#   .tau2/v1.0.1      (commit fc0055d, the grader after them)
# Needs git, uv and network access. Safe to rerun: an existing environment is left as it is.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPO=https://github.com/sierra-research/tau2-bench
LOCK_SHA=62d3a8c4807b89e85703b3c03f2c21048a2da9736ca83d9af6f61adc74ac5314

build() {
  name=$1 commit=$2 dir="$ROOT/.tau2/$1"
  if [ -x "$dir/venv/bin/python" ]; then echo "$name: already built"; return; fi
  mkdir -p "$dir"
  [ -d "$dir/src/.git" ] || git clone --quiet --filter=blob:none "$REPO" "$dir/src"
  git -C "$dir/src" checkout --quiet "$commit"
  actual=$(shasum -a 256 "$dir/src/uv.lock" 2>/dev/null || sha256sum "$dir/src/uv.lock")
  [ "${actual%% *}" = "$LOCK_SHA" ] || { echo "$name: unexpected uv.lock" >&2; exit 1; }
  (cd "$dir/src" && UV_PROJECT_ENVIRONMENT="$dir/venv" uv sync --quiet --frozen --no-install-project --no-build --python 3.12)
  uv pip install --quiet --python "$dir/venv/bin/python" --require-hashes --no-deps --only-binary :all: \
    -r "$ROOT/env/requirements-extra.txt"
  site=$("$dir/venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "$dir/src/src" > "$site/tau2-src.pth"
  "$dir/venv/bin/python" -c "import tau2" && echo "$name: ready ($dir/venv/bin/python)"
}

build pre-v1.0.1 b51a6d69e26f0e94a9173e2e80fe8735a8dff650
build v1.0.1 fc0055dc4e0a316c3f83133267fbd6faaa770992
