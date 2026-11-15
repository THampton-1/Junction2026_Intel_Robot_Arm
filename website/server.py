"""Adapt demo interface: serves the web page, the three camera streams and the inspection state.

Run standalone:
    python website/server.py                 # mock cameras (recorded frames), works anywhere
    python website/server.py --cameras live  # real cameras on the robot laptop (needs OpenCV)
Then open http://localhost:8000 and press Enter.

Only the Python standard library is needed in mock mode. Live mode also needs OpenCV (cv2),
which the anomalib environment already has.

Hooking up the real robot loop later: the loop imports this file and calls
    start_server(on_start=my_cycle)       # my_cycle runs when someone presses Enter
    publish_frame("side", jpeg_bytes)     # if the loop owns the cameras
    publish_state(stage="detect", boxes=[...], score=0.68, verdict="DEFECT", target_view="side")
"""

import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

VIEWS = ["top", "side", "claw"]
THRESHOLD = 0.4781  # PatchCore decision threshold from anomalib test 1

# Live camera devices on the robot laptop (see MASTER_PROJECT_CONTEXT.md, section 2).
# The RealSense exposes several /dev/video nodes; check which one is colour with
#   v4l2-ctl --list-devices   and override with --top /dev/videoNN if 34 is wrong.
LIVE_DEVICES = {"top": "/dev/video34", "side": "/dev/video32", "claw": "/dev/video40"}
# The side and claw cameras share a USB 2.0 hub, so they must stay at 320x240.
LIVE_SIZES = {"top": (640, 480), "side": (320, 240), "claw": (320, 240)}

# Recorded frames used when there is no live camera for a view.
MOCK_FOLDERS = {
    "top": REPO / "anomalib" / "test_2" / "Test 2.1" / "top_view",
    "side": REPO / "anomalib" / "test_2" / "Test 2.1" / "side_view",
    "claw": REPO / "anomalib" / "test_2" / "Test 2.1" / "claw_cam",
}

# ---------------------------------------------------------------------------
# Shared state. Camera threads and the robot loop write here; web requests read.
# A lock stops a request reading a half-written value.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_frames = {}  # view -> (image bytes, mime type)
_state = {
    "run": 0,  # goes up by one each time Enter is pressed; the page uses it to spot a new run
    "stage": "idle",  # idle -> perceive -> detect -> execute -> done
    "boxes": [],  # [{x, y, w, h, score}], all as fractions of the frame (0..1)
    "score": None,
    "threshold": THRESHOLD,
    "verdict": None,  # "GOOD" or "DEFECT"
    "target_view": "side",  # which camera the boxes belong to
    "message": "",
}
_on_start = None  # function run in a background thread when Enter is pressed
_busy = False


def publish_frame(view, image_bytes, mime="image/jpeg"):
    with _lock:
        _frames[view] = (image_bytes, mime)


def publish_state(**fields):
    with _lock:
        _state.update(fields)


def read_state():
    with _lock:
        return dict(_state)


# ---------------------------------------------------------------------------
# Camera sources. Each one runs in its own thread and keeps publishing frames.
# ---------------------------------------------------------------------------
def mock_camera(view, fps=2):
    """Plays the recorded frames for one view as a slow slideshow."""
    paths = sorted(MOCK_FOLDERS[view].glob("*.png"))
    if not paths:
        print(f"[{view}] no recorded frames in {MOCK_FOLDERS[view]}")
        return
    images = [p.read_bytes() for p in paths]
    i = 0
    while True:
        publish_frame(view, images[i], "image/png")
        i = (i + 1) % len(images)
        time.sleep(1 / fps)


def live_camera(view, device):
    """Reads one real camera with OpenCV. Falls back to recorded frames if it can't open."""
    import cv2  # imported here so mock mode works without OpenCV installed

    width, height = LIVE_SIZES[view]
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"[{view}] could not open {device} (in use by Studio?) - showing recorded frames")
        mock_camera(view)
        return
    print(f"[{view}] live from {device}")
    failures = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 30:
                print(f"[{view}] {device} stopped sending frames - showing recorded frames")
                cap.release()
                mock_camera(view)
                return
            time.sleep(0.05)
            continue
        failures = 0
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            publish_frame(view, jpeg.tobytes())


# ---------------------------------------------------------------------------
# Stand-in for the robot loop, so the page can be demoed before the loop exists.
# ---------------------------------------------------------------------------
def simulated_cycle():
    """Fakes one inspection: decoy anomaly peaks plus one strongest peak, then a verdict."""
    publish_state(stage="perceive", boxes=[], score=None, verdict=None, message="Acquiring block")
    time.sleep(2)

    scores = [random.uniform(0.05, 0.42) for _ in range(6)]
    if random.random() < 0.5:
        scores.append(random.uniform(0.5, 0.95))  # a real defect
    else:
        scores.append(random.uniform(0.3, 0.46))  # good block, still has a top peak
    boxes = []
    for s in scores:
        boxes.append(random_box(s, boxes))
    score = max(b["score"] for b in boxes)
    verdict = "DEFECT" if score > THRESHOLD else "GOOD"
    publish_state(stage="detect", boxes=boxes, score=score, verdict=verdict, target_view="side",
                  message="PatchCore on Xe3 iGPU")
    time.sleep(4)

    bowl = "pink" if verdict == "DEFECT" else "green"
    publish_state(stage="execute", message=f"Placing block in {bowl} bowl")
    time.sleep(6)
    publish_state(stage="done", message="Ready")


def random_box(score, existing):
    """A random box that doesn't overlap the boxes already placed (gives up after 50 tries)."""
    for _ in range(50):
        size = random.uniform(0.06, 0.13)
        box = {
            "x": round(random.uniform(0.12, 0.88 - size), 3),
            "y": round(random.uniform(0.12, 0.88 - size * 4 / 3), 3),
            "w": round(size, 3),
            "h": round(size * 4 / 3, 3),  # frames are 4:3, so this makes the box square on screen
            "score": round(score, 4),
        }
        if not any(overlaps(box, other) for other in existing):
            return box
    return box


def overlaps(a, b, gap=0.03):
    return (a["x"] < b["x"] + b["w"] + gap and b["x"] < a["x"] + a["w"] + gap
            and a["y"] < b["y"] + b["h"] + gap and b["y"] < a["y"] + a["h"] + gap)


def _run_cycle():
    global _busy
    try:
        _on_start()
    except Exception as error:  # show the problem on the page instead of silently dying
        publish_state(stage="done", message=f"Error: {error}")
        raise
    finally:
        _busy = False


# ---------------------------------------------------------------------------
# Chatbot ("Chat with Adapt"). Uses the Gemini API with the standard library only.
# The key is read from website/gemini_key.txt (or GEMINI_API_KEY), so it never reaches the page.
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# What the chatbot knows about the project. Edit these facts to match the final system.
CHAT_FACTS = (
    "You are Adapt, the assistant on the Adapt live inspection demo page, built by a team at "
    "Junction 2026 for the Intel Physical AI Challenge. Facts about the system: an SO-101 robot "
    "arm inspects Duplo blocks. Three cameras watch the cell: an overhead RealSense, a side "
    "webcam and a claw camera. The arm rotates each block 90 degrees at a time so the side camera "
    "sees all four sides. A PatchCore anomaly model (Anomalib, optimised with OpenVINO) runs on "
    "the Intel Xe3 integrated GPU of an Intel Core Ultra laptop in about 3.78 ms and scores each "
    "view; a peak score above the threshold means DEFECT. The defect is a small black mark on one "
    "side of the block. A learned ACT policy, trained from teleoperated demonstrations, moves the "
    "arm and places the block in the pink bowl (defect) or the green bowl (good). "
    "Rules: answer in plain text with no markdown, in at most three short sentences. "
    "If you do not know something, say so instead of guessing."
)


def read_gemini_key():
    """The key comes from website/gemini_key.txt, or the GEMINI_API_KEY environment variable."""
    key_file = HERE / "gemini_key.txt"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key and "PASTE" not in key:
            return key
    return os.environ.get("GEMINI_API_KEY", "").strip()


def chat_reply(messages):
    """Sends the recent conversation to Gemini and returns its answer as plain text."""
    key = read_gemini_key()
    if not key:
        return "Chat is offline. Paste your Gemini key into website/gemini_key.txt."

    contents = []
    for message in messages[-9:]:  # recent turns only: keeps token use small
        if not isinstance(message, dict):
            continue
        text = str(message.get("text", "")).strip()[:600]
        role = "model" if message.get("role") == "model" else "user"
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    while contents and contents[0]["role"] == "model":  # Gemini expects the user to speak first
        contents.pop(0)
    if not contents:
        return "Type a question first."

    state = read_state()
    live = (f" Live state of the cell right now: stage={state['stage']}, peak score={state['score']}, "
            f"verdict={state['verdict']}, threshold={state['threshold']}, status={state['message']!r}.")
    body = {
        "system_instruction": {"parts": [{"text": CHAT_FACTS + live}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.4},
    }
    request = urllib.request.Request(
        GEMINI_URL.format(model=GEMINI_MODEL),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
        parts = data["candidates"][0]["content"]["parts"]
        text = " ".join(part.get("text", "") for part in parts if not part.get("thought"))
        return " ".join(text.split()) or "I have no answer for that one."
    except urllib.error.HTTPError as error:
        print(f"[chat] Gemini returned HTTP {error.code}: {error.read()[:300]!r}")
        if error.code == 429:
            return "I've hit the free usage limit. Try again in a minute."
        return f"Gemini refused the request (HTTP {error.code}). Check GEMINI_API_KEY and GEMINI_MODEL."
    except (KeyError, IndexError, TypeError):
        return "I have no answer for that one."
    except Exception as error:
        print(f"[chat] could not reach Gemini: {error}")
        return "I can't reach Gemini right now. Check the internet connection."


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path  # drop any "?..." query string
        if path in ("/", "/index.html"):
            self.send_bytes((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send_bytes(json.dumps(read_state()).encode(), "application/json")
        elif path.startswith("/snapshot/"):
            view = path.split("/")[-1]
            with _lock:
                frame = _frames.get(view)
            if frame is None:
                self.send_error(404, "no frame yet")
            else:
                self.send_bytes(frame[0], frame[1])
        elif path.startswith("/stream/"):
            self.stream(path.split("/")[-1])
        elif path == "/favicon.ico":
            self.send_response(204)  # no icon; avoids a 404 in the browser console
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        global _busy
        if urlsplit(self.path).path == "/api/chat":
            self.chat()
            return
        if urlsplit(self.path).path != "/api/start":
            self.send_error(404)
            return
        if _busy:
            self.send_bytes(b'{"started": false}', "application/json", status=409)
            return
        _busy = True
        publish_state(run=read_state()["run"] + 1, stage="perceive", boxes=[], score=None,
                      verdict=None, message="Starting")
        threading.Thread(target=_run_cycle, daemon=True).start()
        self.send_bytes(b'{"started": true}', "application/json")

    def chat(self):
        """POST /api/chat with {"messages": [{"role": "user" or "model", "text": "..."}]}."""
        length = min(int(self.headers.get("Content-Length") or 0), 20000)
        try:
            messages = json.loads(self.rfile.read(length) or b"{}").get("messages", [])
        except (ValueError, AttributeError):
            messages = []
        if not isinstance(messages, list):
            messages = []
        reply = chat_reply(messages)
        self.send_bytes(json.dumps({"reply": reply}).encode(), "application/json")

    def send_bytes(self, body, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def stream(self, view):
        """MJPEG-style stream: one HTTP response that keeps replacing the image.
        Browsers show this in a plain <img> tag as live video."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        try:
            while True:
                with _lock:
                    frame = _frames.get(view)
                if frame is not None and frame is not last:
                    image, mime = frame
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(f"Content-Type: {mime}\r\nContent-Length: {len(image)}\r\n\r\n".encode())
                    self.wfile.write(image)
                    self.wfile.write(b"\r\n")
                    last = frame
                time.sleep(1 / 20)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser closed the page

    def log_message(self, *args):
        pass  # keep the terminal quiet; the streams would otherwise log constantly


def start_server(port=8000, on_start=simulated_cycle):
    """Starts the web server in a background thread and returns immediately."""
    global _on_start
    _on_start = on_start
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Adapt interface on http://localhost:{port}")
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", choices=["mock", "live"], default="mock")
    parser.add_argument("--port", type=int, default=8000)
    for view in VIEWS:
        parser.add_argument(f"--{view}", default=LIVE_DEVICES[view], help=f"{view} camera device")
    args = parser.parse_args()

    for view in VIEWS:
        if args.cameras == "live":
            target, target_args = live_camera, (view, getattr(args, view))
        else:
            target, target_args = mock_camera, (view,)
        threading.Thread(target=target, args=target_args, daemon=True).start()

    start_server(args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
