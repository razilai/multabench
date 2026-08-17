#!/bin/bash
# Usage: source init.sh
# Installs uv, creates .venv with prebuilt Python 3.11, installs deps.

ENV_DIR=".venv"

if ! command -v uv &>/dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "🌀 Creating venv with Python 3.11 (uv fetches prebuilt CPython)"
uv venv -p 3.11 "$ENV_DIR" || return 1

echo "🚀 Activating venv ($ENV_DIR)"
source "$ENV_DIR/bin/activate" || return 1

export UV_HTTP_TIMEOUT=120
if [ -f "requirements.txt" ]; then
    echo "📄 Installing dependencies from requirements.txt"
    uv pip install -r requirements.txt || return 1
else
    echo "⚠️ requirements.txt not found; skipping dependency install."
fi

echo "🛠 Adding repo root to Python path via .pth"
SITE_PACKAGES=$("$ENV_DIR/bin/python" -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)" > "$SITE_PACKAGES/tabstar_paper.pth"

echo "🎉 Setup completed! Venv active. New shell: source $ENV_DIR/bin/activate"
