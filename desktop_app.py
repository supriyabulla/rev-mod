"""
desktop_app.py
---------------
macOS desktop app launcher.
Starts the FastAPI server in a background thread, then opens
a native macOS window (WKWebView) pointing at localhost.

Run with:  python desktop_app.py
"""

import sys
import os
import time
import threading
import subprocess
from pathlib import Path

# Add parent to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
import webview


SERVER_PORT = 7331
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"


def start_server():
    """Start the FastAPI server in a daemon thread."""
    from app.server import app as fastapi_app
    config = uvicorn.Config(
        fastapi_app,
        host="127.0.0.1",
        port=SERVER_PORT,
        log_level="warning",   # suppress access logs in desktop mode
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    server.run()


def wait_for_server(timeout: int = 15) -> bool:
    """Poll until the server is ready."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{SERVER_URL}/api/status", timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


class AppAPI:
    """
    Exposed to JavaScript via window.pywebview.api.*
    Provides native macOS capabilities: file picker, drag-and-drop.
    """

    def open_file_dialog(self):
        """Open native macOS file picker for PDF selection."""
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("PDF Files (*.pdf)",),
        )
        if result and len(result) > 0:
            return result[0]
        return None

    def get_app_version(self):
        return "1.0.0"

    def quit_app(self):
        webview.windows[0].destroy()


def main():
    print("🚀 Starting Study Assistant...")

    # Start API server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    print("⏳ Starting local AI server...")
    if not wait_for_server(timeout=20):
        print("❌ Server failed to start")
        sys.exit(1)

    print("✅ Server ready — opening window")

    # Create native macOS window
    api = AppAPI()
    window = webview.create_window(
        title="Study Assistant",
        url=SERVER_URL,
        js_api=api,
        width=1200,
        height=820,
        min_size=(900, 680),
        resizable=True,
        shadow=True,
        easy_drag=False,
        frameless=False,
        text_select=False,
        background_color="#0F1117",
    )

    # Start pywebview (blocks until window closed)
    webview.start(
        debug=False,
        # Use WKWebView on macOS (best performance)
        gui="cocoa",
    )


if __name__ == "__main__":
    main()
