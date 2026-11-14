#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Helper script to automatically stage, commit, and push Anomalib artifacts to Git

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MSG="${1:-feat(anomalib): sync updated test artifacts and benchmark metrics}"

echo "=================================================="
echo "  ANOMALIB AUTO-SYNC TO GIT (github.com)"
echo "=================================================="

# Ensure no binary files larger than 10MB are staged accidentally
echo "Checking for oversized binary files (>10MB)..."
find anomalib/ -type f -size +10M 2>/dev/null | while read -r file; do
    echo "ERROR: File $file exceeds 10MB! Aborting push to prevent git bloat."
    exit 1
done

echo "Staging anomalib/ artifacts and scripts..."
git add anomalib/ scripts/ MASTER_PROJECT_CONTEXT.md .gitignore

if git diff --staged --quiet; then
    echo "Working tree is clean. Nothing new to commit."
    exit 0
fi

echo "Committing with message: '$MSG'..."
git commit -m "$MSG"

echo "Pushing to origin main..."
git push origin main

echo "=================================================="
echo "✓ Sync complete! Repo updated successfully."
echo "=================================================="
