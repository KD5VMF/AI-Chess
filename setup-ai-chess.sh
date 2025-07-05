#!/bin/bash

# === CONFIG ===
APPDIR=~/MyApps/AI-Chess/AI-Chess2
REPO=https://github.com/KD5VMF/AI-Chess.git
VENV=~/envChess
LAUNCHER=~/start-ai-chess.sh

echo "🧠 Setting up AI Chess Trainer..."

# Step 1: Clone repo or pull latest
if [ ! -d "$APPDIR" ]; then
    echo "📥 Cloning repository..."
    mkdir -p "$(dirname "$APPDIR")" || { echo "❌ Failed to create base folder"; exit 1; }
    git clone "$REPO" "$(dirname "$APPDIR")" || { echo "❌ Git clone failed"; exit 1; }
else
    echo "🔄 Pulling latest code..."
    git -C "$(dirname "$APPDIR")" pull || { echo "❌ Git pull failed"; exit 1; }
fi

# Step 2: Install system dependencies
echo "🛠 Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git gcc || {
    echo "❌ Failed to install required system packages"
    exit 1
}

# Step 3: Create Python virtual environment
if [ ! -d "$VENV" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv "$VENV" || { echo "❌ Failed to create virtual environment"; exit 1; }
fi

# Step 4: Activate and install Python packages
echo "📦 Activating environment and installing Python dependencies..."
source "$VENV/bin/activate" || { echo "❌ Failed to activate virtual environment"; exit 1; }
pip install --upgrade pip setuptools wheel || { echo "❌ Failed to upgrade pip"; exit 1; }

pip install psutil python-chess numpy torch matplotlib pandas screen || {
    echo "❌ One or more Python packages failed to install"
    exit 1
}

# Step 5: Create launcher script
echo "🧷 Creating launcher at $LAUNCHER..."
cat <<EOF > "$LAUNCHER"
#!/bin/bash
screen -S ai-chess -dm bash -c '
source $VENV/bin/activate
cd $APPDIR
python AI-Chess.py
'
EOF

chmod +x "$LAUNCHER" || { echo "❌ Failed to make launcher executable"; exit 1; }

# === Final Output ===
echo ""
echo "✅ AI Chess Trainer is ready!"
echo ""
echo "👉 To start the trainer:      $LAUNCHER"
echo "🔁 To reattach the session:   screen -r ai-chess"
echo "🧹 To stop the session:       screen -X -S ai-chess quit"
echo ""
echo "🗂 Save files will appear in: $APPDIR"
