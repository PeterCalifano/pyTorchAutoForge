#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-site}"
BASE_URL="${PTAF_DOC_BASE_URL:-https://petercalifano.github.io/pyTorchAutoForge}"
SWITCHER_JSON_URL="${PTAF_DOC_SWITCHER_JSON_URL:-${BASE_URL}/_static/switcher.json}"

cd "${REPO_ROOT}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/_static"

build_docs_() {
    local source_root_="$1"
    local output_name_="$2"
    local version_name_="$3"

    if [ ! -f "${source_root_}/doc/conf.py" ]; then
        echo "Skipping ${version_name_}: doc/conf.py not found"
        return 0
    fi

    if [ "${version_name_}" != "stable" ] && [ ! -f "${source_root_}/doc/_static/switcher.json" ]; then
        echo "Skipping ${version_name_}: versioned docs config not found"
        return 0
    fi

    echo "Building docs ${version_name_} -> ${OUTPUT_DIR}/${output_name_}"
    if ! PTAF_DOC_BASE_URL="${BASE_URL}/${output_name_}/" \
        PTAF_DOC_VERSION="${version_name_}" \
        PTAF_DOC_SWITCHER_JSON_URL="${SWITCHER_JSON_URL}" \
            python -m sphinx -b html "${source_root_}/doc" "${OUTPUT_DIR}/${output_name_}"; then
        echo "Skipping ${version_name_}: Sphinx build failed"
        rm -rf "${OUTPUT_DIR:?}/${output_name_}"
    fi
}

build_docs_ "${REPO_ROOT}" "stable" "stable"

TEMP_WORKTREE_ROOT="$(mktemp -d)"
cleanup_() {
    find "${TEMP_WORKTREE_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null |
        while IFS= read -r -d '' worktree_dir_; do
            git worktree remove --force "${worktree_dir_}" >/dev/null 2>&1 || true
        done
    rm -rf "${TEMP_WORKTREE_ROOT}"
}
trap cleanup_ EXIT

while IFS= read -r tag_name_; do
    [ -n "${tag_name_}" ] || continue

    output_name_="${tag_name_//\//-}"
    worktree_dir_="${TEMP_WORKTREE_ROOT}/${output_name_}"

    git worktree add --detach --quiet "${worktree_dir_}" "${tag_name_}"
    build_docs_ "${worktree_dir_}" "${output_name_}" "${tag_name_}"
    git worktree remove --force "${worktree_dir_}" >/dev/null
done < <(git tag --list 'v*' --sort=-v:refname)

python - "${OUTPUT_DIR}" "${BASE_URL}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

output_dir_ = Path(sys.argv[1])
base_url_ = sys.argv[2].rstrip("/")

entries_: list[dict[str, str]] = []

stable_dir_ = output_dir_ / "stable"
if (stable_dir_ / "index.html").exists():
    entries_.append(
        {
            "name": "stable",
            "version": "stable",
            "url": f"{base_url_}/stable/",
        }
    )

for version_dir_ in sorted(output_dir_.iterdir(), reverse=True):
    if not version_dir_.is_dir():
        continue
    version_name_ = version_dir_.name
    if version_name_ in {"_static", "stable"}:
        continue
    if not (version_dir_ / "index.html").exists():
        continue
    entries_.append(
        {
            "name": version_name_,
            "version": version_name_,
            "url": f"{base_url_}/{version_name_}/",
        }
    )

(output_dir_ / "_static").mkdir(parents=True, exist_ok=True)
(output_dir_ / "_static" / "switcher.json").write_text(
    json.dumps(entries_, indent=2) + "\n",
    encoding="utf-8",
)

(output_dir_ / ".nojekyll").write_text("", encoding="utf-8")
(output_dir_ / "index.html").write_text(
    """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=stable/">
    <link rel="canonical" href="stable/">
    <title>pyTorchAutoForge Documentation</title>
  </head>
  <body>
    <p><a href="stable/">Open stable documentation</a></p>
  </body>
</html>
""",
    encoding="utf-8",
)
PY
