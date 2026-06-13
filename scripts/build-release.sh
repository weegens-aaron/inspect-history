#!/usr/bin/env bash
#
# build-release.sh — Build the clean inspect-history release zip from an
# explicit allowlist.
#
# What it does (deterministic, idempotent):
#   1. Reads __version__ from __init__.py (single source of truth, no dupes).
#   2. Cleans staging/ and dist/.
#   3. Copies ONLY the allowlisted runtime paths into staging/inspect_history/.
#   4. Zips staging/ so the archive's single top-level entry is inspect_history/.
#   5. Writes BOTH a stable name (dist/inspect-history.zip — enables the
#      /releases/latest/download/inspect-history.zip URL) and a versioned name
#      (dist/inspect-history-v<version>.zip).
#   6. Writes a SHA256 checksum file alongside each zip so users — and the
#      published GitHub Release — can verify download integrity.
#   7. Self-checks: extracts the stable zip to a temp dir and verifies the
#      package imports. A missing runtime file => failure => the allowlist is
#      incomplete and the build fails loudly.
#
# Design: allowlist COPY (not `git archive`) so the clean subset is guaranteed
# regardless of git tracking state. The file list lives in ONE array below.
#
# NOTE on naming: the zip's top-level folder is the IMPORT name
# `inspect_history` (underscore — a valid Python package identifier), while the
# zip FILE and the GitHub asset use the repo name `inspect-history` (hyphen).
# Code Puppy resolves the plugin's relative imports under either directory name
# at runtime; the underscore folder simply lets the self-check `import` cleanly.

set -euo pipefail

# --- Locate the repo root (the dir that holds __init__.py), relative to this
#     script, so the build works no matter the caller's CWD. -----------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# --- The allowlist: SHIP = runtime .py files + README.md + LICENSE. Anything
#     not named here is excluded by construction (fail-closed). LICENSE ships
#     so the MIT terms travel with the artifact. -------------------------------
ALLOWLIST=(
  "__init__.py"
  "register_callbacks.py"
  "inspect_model.py"
  "inspect_render.py"
  "inspect_menu.py"
  "README.md"
  "LICENSE"
)

PKG_NAME="inspect_history"
PYTHON="${PYTHON:-python3}"
STAGING_DIR="${REPO_ROOT}/staging"
DIST_DIR="${REPO_ROOT}/dist"
STABLE_ZIP="${DIST_DIR}/inspect-history.zip"

# --- 1. Read the single-source __version__ from __init__.py (no Python import
#        needed — just slice the quoted value out of the one assignment). -----
read_version() {
  local line
  line="$(grep -E '^__version__[[:space:]]*=' "${REPO_ROOT}/__init__.py" | head -n1)"
  if [[ -z "${line}" ]]; then
    echo "ERROR: could not find __version__ in __init__.py" >&2
    exit 1
  fi
  local ver
  ver="$(printf '%s\n' "${line}" | sed -E 's/^[^"]*"([^"]*)".*$/\1/')"
  if [[ -z "${ver}" || "${ver}" == "${line}" ]]; then
    echo "ERROR: could not parse a quoted version from: ${line}" >&2
    exit 1
  fi
  printf '%s\n' "${ver}"
}

# --- Cross-platform SHA256: macOS ships `shasum`, most Linux ships
#     `sha256sum`. The checksum is written from inside dist/ so it references
#     the BARE filename (what `shasum -a 256 -c` / `sha256sum -c` expect). ----
sha256_in_dist() {
  local file="$1"  # bare filename, relative to DIST_DIR
  (
    cd "${DIST_DIR}"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "${file}" > "${file}.sha256"
    elif command -v shasum >/dev/null 2>&1; then
      shasum -a 256 "${file}" > "${file}.sha256"
    else
      echo "ERROR: neither sha256sum nor shasum found — cannot checksum ${file}" >&2
      exit 1
    fi
  )
}

VERSION="$(read_version)"
VERSIONED_ZIP="${DIST_DIR}/inspect-history-v${VERSION}.zip"

echo "==> inspect-history release build (v${VERSION})"

# --- 2. Clean staging/ and dist/ (idempotent re-runs). ----------------------
echo "==> Cleaning staging/ and dist/"
rm -rf "${STAGING_DIR}" "${DIST_DIR}"
mkdir -p "${STAGING_DIR}/${PKG_NAME}" "${DIST_DIR}"

# --- 3. Copy ONLY the allowlisted paths into staging/inspect_history/. -------
echo "==> Copying ${#ALLOWLIST[@]} allowlisted paths into staging/${PKG_NAME}/"
for path in "${ALLOWLIST[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
    echo "ERROR: allowlisted path is missing from the repo: ${path}" >&2
    exit 1
  fi
  cp -f "${REPO_ROOT}/${path}" "${STAGING_DIR}/${PKG_NAME}/"
  echo "    + ${path}"
done

# --- 4 & 5. Zip staging/ so the single top-level entry is inspect_history/. --
echo "==> Building ${STABLE_ZIP}"
(
  cd "${STAGING_DIR}"
  # -X strips extra file attributes (no .DS_Store-style noise); -r recursive.
  zip -X -r -q "${STABLE_ZIP}" "${PKG_NAME}"
)
cp -f "${STABLE_ZIP}" "${VERSIONED_ZIP}"
echo "    wrote $(basename "${STABLE_ZIP}") and $(basename "${VERSIONED_ZIP}")"

# --- 6. Write SHA256 checksum files alongside the zips. ---------------------
echo "==> Writing SHA256 checksums"
sha256_in_dist "$(basename "${STABLE_ZIP}")"
sha256_in_dist "$(basename "${VERSIONED_ZIP}")"
echo "    wrote $(basename "${STABLE_ZIP}").sha256 and $(basename "${VERSIONED_ZIP}").sha256"

# --- 7. Self-check: extract to a temp dir and verify the package. -----------
#        The full /inspect entry point needs code_puppy + prompt_toolkit, which
#        may be absent in a bare CI shell. So we ALWAYS byte-compile every
#        shipped .py (catches syntax errors) and import the dependency-free
#        core modules (inspect_model + inspect_render — proves the package and
#        its relative imports resolve). If code_puppy IS importable we go the
#        whole way and import register_callbacks too.
echo "==> Self-check: extracting and verifying ${PKG_NAME}"
TMP_CHECK="$(mktemp -d)"
cleanup() { rm -rf "${TMP_CHECK}"; }
trap cleanup EXIT

unzip -q "${STABLE_ZIP}" -d "${TMP_CHECK}"

"${PYTHON}" - "${TMP_CHECK}" "${PKG_NAME}" <<'PYCHECK'
import importlib, py_compile, sys
from pathlib import Path

tmp, pkg = sys.argv[1], sys.argv[2]
pkg_dir = Path(tmp) / pkg
sys.path.insert(0, tmp)

# (a) byte-compile every shipped .py — fails loudly on any syntax error.
for py in sorted(pkg_dir.glob("*.py")):
    py_compile.compile(str(py), doraise=True)
print(f"    byte-compile OK: {len(list(pkg_dir.glob('*.py')))} file(s)")

# (b) import the dependency-free core modules — proves the package + its
#     relative imports resolve from the zip.
for mod in (f"{pkg}.inspect_model", f"{pkg}.inspect_render"):
    importlib.import_module(mod)
    print(f"    import OK: {mod}")

# (c) full entry-point import only if code_puppy is available.
try:
    import code_puppy  # noqa: F401
except ImportError:
    print("    skip: code_puppy not installed — entry-point import not exercised")
else:
    importlib.import_module(f"{pkg}.register_callbacks")
    print(f"    import OK: {pkg}.register_callbacks")
PYCHECK

# --- Report the archive contents so the build is auditable at a glance. ------
echo "==> Archive contents:"
unzip -l "${STABLE_ZIP}"

echo "==> Done. Release artifacts in dist/:"
ls -1 "${DIST_DIR}"

# --- Release reminder: upload the zips AND their .sha256 sidecars as assets. -
echo
echo "==> To publish: upload BOTH the zips AND their .sha256 files, e.g."
echo "      gh release create v${VERSION} \\"
echo "        ${DIST_DIR}/inspect-history*.zip ${DIST_DIR}/inspect-history*.zip.sha256"
echo "    (the .sha256 assets are what the README's verify step downloads)"
