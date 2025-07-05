#!/bin/bash

# === CONFIG ===
APPDIR=~/MyApps/AI-Chess/AI-Chess2
REPO=https://github.com/KD5VMF/AI-Chess.git
VENV=~/envChess
LAUNCHER=~/start-ai-chess.sh

echo "🧠 Setting up AI Chess Trainer..."

# Clone repo if not present
if [ ! -d "$APPDIR" ]; then
    echo "📥 Cloning repository..."
    mkdir -p "$(dirname "$APPDIR")"
    git clone "$REPO" "$(dirname "$APPDIR")"
else
    echo "🔄 Pulling latest code..."
    git -C "$(dirname "$APPDIR")" pull
fi

# Install Python 3 venv if not present
sudo apt-get install -y python3-venv python3-pip git gcc

# Create virtual environment
if [ ! -d "$VENV" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv "$VENV"
fi

# Activate and install dependencies
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

echo "📦 Installing required packages..."
pip install psutil python-chess numpy torch matplotlib pandas screen

# Create launcher
echo "🧷 Creating launcher script at $LAUNCHER..."
cat <<EOF > "$LAUNCHER"
#!/bin/bash
screen -S ai-chess -dm bash -c '
source $VENV/bin/activate
cd $APPDIR
python AI-Chess.py
'
EOF
chmod +x "$LAUNCHER"

echo "✅ Setup complete!"
echo "▶ Run your AI Chess: $LAUNCHER"
echo "🔁 Reattach screen session: screen -r ai-chess"
