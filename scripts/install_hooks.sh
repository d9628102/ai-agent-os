#!/usr/bin/env bash
# One-time setup: point git at this repo's tracked githooks/ directory
# instead of the untracked (and never auto-installed) .git/hooks/.
# Run once per clone: python3/bash scripts/install_hooks.sh
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
git -C "${repo_root}" config core.hooksPath githooks
chmod +x "${repo_root}/githooks/pre-push"
echo "core.hooksPath -> githooks/ 已設定，pre-push 程式碼審查閘門生效。"
