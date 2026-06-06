#!/usr/bin/env bash
# setup_desktop.sh — One-command setup for Study Assistant macOS Desktop App
set -e

CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"
BOLD="\033[1m"; RESET="\033[0m"

echo ""
echo -e "${CYAN}${BOLD}  ╔══════════════════════════════════════════════════════╗"
echo -e "  ║   📚 Study Assistant v2 — Desktop App Setup         ║"
echo -e "  ╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

echo -e "${CYAN}[1/5] Checking Python 3.9+...${RESET}"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}  ❌ Python 3 not found. Install: brew install python${RESET}"; exit 1
fi
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}  ✅ Python ${PYVER}${RESET}"

echo -e "\n${CYAN}[2/5] Checking Homebrew...${RESET}"
if ! command -v brew &>/dev/null; then
    echo -e "${YELLOW}  Installing Homebrew...${RESET}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
echo -e "${GREEN}  ✅ Homebrew ready${RESET}"

echo -e "\n${CYAN}[3/5] Setting up Ollama (local AI)...${RESET}"
if ! command -v ollama &>/dev/null; then
    echo -e "${YELLOW}  Installing Ollama...${RESET}"
    brew install ollama
fi
echo -e "${GREEN}  ✅ Ollama installed${RESET}"

if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo -e "${YELLOW}  Starting Ollama...${RESET}"
    ollama serve &>/dev/null & sleep 4
fi

echo ""
echo "  Recommended models for Apple Silicon:"
echo "    1) llama3.2  (3B, 2GB)  ← BEST for M1/M2"
echo "    2) mistral   (7B, 4GB)  ← Best quality, needs M2 Pro+"
echo "    3) phi3      (3.8B)     ← Fast & efficient"
echo "    4) gemma2:2b (2B)       ← Fastest"
echo "    5) Skip (already installed)"
echo ""
read -p "  Choose [1-5, default=1]: " mc; mc=${mc:-1}
case $mc in
    1) M="llama3.2" ;; 2) M="mistral" ;;
    3) M="phi3"     ;; 4) M="gemma2:2b" ;; *) M="" ;;
esac
[ -n "$M" ] && { echo -e "${YELLOW}  Pulling ${M}...${RESET}"; ollama pull "$M"; }
echo -e "${GREEN}  ✅ AI model ready${RESET}"

echo -e "\n${CYAN}[4/5] Installing Python packages...${RESET}"
[ ! -d "venv" ] && python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet
pip install PyMuPDF fastapi "uvicorn[standard]" python-multipart pywebview pydantic --quiet
echo -e "${GREEN}  ✅ All packages installed${RESET}"

echo -e "\n${CYAN}[5/5] Verifying...${RESET}"
python3 -c "import fitz; print('  ✅ PyMuPDF:', fitz.__version__)" 2>/dev/null || echo -e "${RED}  ❌ PyMuPDF${RESET}"
python3 -c "import fastapi; print('  ✅ FastAPI:', fastapi.__version__)" 2>/dev/null || echo -e "${RED}  ❌ FastAPI${RESET}"
python3 -c "import webview; print('  ✅ pywebview:', webview.__version__)" 2>/dev/null || echo -e "${RED}  ❌ pywebview${RESET}"

echo ""
echo -e "${GREEN}${BOLD}  ══════════════════════════════════════════════════════"
echo -e "  ✅ Setup Complete!"
echo -e "  ══════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${BOLD}In a separate terminal, keep Ollama running:${RESET}"
echo -e "    ollama serve"
echo ""
echo -e "  ${BOLD}Launch the desktop app:${RESET}"
echo -e "    source venv/bin/activate"
echo -e "    python desktop_app.py"
echo ""
