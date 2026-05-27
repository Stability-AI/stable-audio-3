#!/usr/bin/env python3
"""
insert_audio.py — CLI tool to insert audio files into Ableton Live.

Usage:
    python insert_audio.py                      # insert latest file in ~/Desktop
    python insert_audio.py /path/to/file.wav    # insert specific file
    python insert_audio.py --watch ~/renders    # watch folder, auto-insert on new file
    python insert_audio.py --ping               # check if Ableton is connected

Options:
    --dir DIR       Directory to search for latest file (default: ~/Desktop)
    --ext EXT       File extension filter, e.g. wav, aiff, mp3 (default: wav aiff mp3 flac)
    --watch         Watch mode: monitor DIR and auto-insert new files as they appear
    --ping          Just check if the Ableton script is running
"""

import sys
import os
import json
import socket
import argparse
import time

HOST = "127.0.0.1"
PORT = 9129
TIMEOUT = 5.0
AUDIO_EXTENSIONS = {".wav", ".aiff", ".aif", ".mp3", ".flac", ".ogg", ".m4a"}


# ─────────────────────────────────────────────
# Socket communication
# ─────────────────────────────────────────────

def send_command(cmd: dict) -> dict:
    """Send a JSON command to the Ableton Remote Script and return the response."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        s.connect((HOST, PORT))
        s.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        response = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            response += chunk
            if b"\n" in chunk:
                break
        s.close()
        return json.loads(response.decode("utf-8").strip())
    except ConnectionRefusedError:
        return {"ok": False, "error": "Cannot connect to Ableton. Make sure:\n"
                "  1. Ableton Live is open\n"
                "  2. AudioInserter is selected in Preferences → MIDI → Control Surface"}
    except socket.timeout:
        return {"ok": False, "error": "Ableton timed out — is a project open?"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def ping():
    result = send_command({"action": "ping"})
    if result.get("ok"):
        print("✓ Ableton is connected and ready")
        return True
    else:
        print("✗ " + result.get("error", "Unknown error"))
        return False


def insert_file(file_path: str) -> bool:
    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        print(f"✗ File not found: {abs_path}")
        return False

    print(f"→ Inserting: {os.path.basename(abs_path)}")
    track_name = os.environ.get("_SA3_LAST_PROMPT", "")
    result = send_command({"action": "insert_audio", "file_path": abs_path, "track_name": track_name})

    if result.get("ok"):
        msg = result.get("message", "Done")
        print(f"✓ {msg}")
        return True
    else:
        print(f"✗ {result.get('error', 'Unknown error')}")
        return False


# ─────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────

def find_latest_audio(directory: str, extensions: set) -> str | None:
    """Return the most recently modified audio file in the given directory."""
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        print(f"✗ Directory not found: {directory}")
        return None

    candidates = []
    for f in os.listdir(directory):
        ext = os.path.splitext(f)[1].lower()
        if ext in extensions:
            full = os.path.join(directory, f)
            candidates.append((os.path.getmtime(full), full))

    if not candidates:
        exts = " ".join(sorted(extensions))
        print(f"✗ No audio files found in {directory}\n  (looking for: {exts})")
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


# ─────────────────────────────────────────────
# Watch mode
# ─────────────────────────────────────────────

def watch_directory(directory: str, extensions: set):
    """Monitor a directory and auto-insert new audio files as they appear."""
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        print(f"✗ Directory not found: {directory}")
        sys.exit(1)

    print(f"👁  Watching {directory} for new audio files... (Ctrl+C to stop)\n")

    # Record existing files so we don't re-insert them on start
    seen = set()
    for f in os.listdir(directory):
        if os.path.splitext(f)[1].lower() in extensions:
            seen.add(os.path.join(directory, f))

    try:
        while True:
            time.sleep(0.75)
            for f in os.listdir(directory):
                ext = os.path.splitext(f)[1].lower()
                if ext not in extensions:
                    continue
                full = os.path.join(directory, f)
                if full not in seen:
                    seen.add(full)
                    # Short wait to ensure file write is complete
                    time.sleep(0.3)
                    print(f"\n[{time.strftime('%H:%M:%S')}] New file detected")
                    insert_file(full)
    except KeyboardInterrupt:
        print("\n\nStopped watching.")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Insert audio files into Ableton Live at the current playhead position.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file", nargs="?", help="Audio file to insert (omit to use latest file in --dir)"
    )
    parser.add_argument(
        "--dir", default="~/Desktop", help="Directory to search for latest audio file"
    )
    parser.add_argument(
        "--ext", nargs="+", default=None,
        help="File extensions to consider, e.g. --ext wav aiff"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch --dir and auto-insert new files as they appear"
    )
    parser.add_argument(
        "--ping", action="store_true",
        help="Check if Ableton is running and the script is active"
    )

    args = parser.parse_args()

    # Build extension set
    if args.ext:
        extensions = {"." + e.lstrip(".").lower() for e in args.ext}
    else:
        extensions = AUDIO_EXTENSIONS

    # ── Ping ──
    if args.ping:
        sys.exit(0 if ping() else 1)

    # ── Watch mode ──
    if args.watch:
        watch_directory(args.dir, extensions)
        return

    # ── Single insert ──
    if args.file:
        file_path = args.file
    else:
        file_path = find_latest_audio(args.dir, extensions)
        if not file_path:
            sys.exit(1)
        print(f"Latest file: {os.path.basename(file_path)}")

    success = insert_file(file_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
