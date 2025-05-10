#!/bin/bash
set -e

echo "🔁 Moving module directories into 'core/'..."

mkdir -p core/execution core/planner core/embeddings core/entity

# Move each module's files to its target location
mv modules/execution/*.py core/execution/
mv modules/planning/*.py core/planner/
mv modules/embedding/*.py core/embeddings/
mv modules/resolution/*.py core/entity/

echo "✅ Move complete."

# Optional cleanup
rmdir modules/execution modules/planning modules/embedding modules/resolution modules 2>/dev/null || true
