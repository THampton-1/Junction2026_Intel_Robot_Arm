# Adapt demo interface

A retro web page for the live demo. Before you press Enter, the hero shows a blue-tinted still picture. After Enter, it plays the three live
camera feeds (switching every 3 s). An aim dot orbits the frame, then hops between candidate anomaly
regions from lowest to highest score and locks onto the strongest one with the GOOD / DEFECT verdict.

```bash
python website/server.py                    # recorded frames, runs anywhere (standard library only)
python website/server.py --cameras live     # real cameras, on the robot laptop (needs OpenCV)
# open http://localhost:8000, press Enter.  /?autostart starts a run on page load.
```

| File | What it does |
|---|---|
| `server.py` | Web server: the page, `/stream/{top,side,claw}` live video, `/snapshot/<view>`, `/api/state`, `POST /api/start` |
| `index.html` | The whole page: layout, chat, camera cycling, reticle animation. No libraries. |

## Before demo day

- **Camera ownership.** Only one program can open a camera. If Physical AI Studio (or the robot loop)
  has them open, `--cameras live` prints a warning and falls back to recorded frames for that view.
- **RealSense node.** The D415 exposes several `/dev/video*` nodes. Run `v4l2-ctl --list-devices`
  and pass the colour one with `--top /dev/videoNN` if 34 is wrong.
- **The verdict is still simulated.** `simulated_cycle()` in `server.py` makes up the candidate
  boxes. Replace it with the real loop (below).

## Hooking up the real robot loop

```python
import server
server.start_server(on_start=run_one_block)   # run_one_block() is called when Enter is pressed

def run_one_block():
    server.publish_state(stage="perceive")
    ...
    server.publish_state(stage="detect", target_view="side", boxes=boxes,
                         score=score, verdict="DEFECT" if score > server.THRESHOLD else "GOOD")
    ...
    server.publish_state(stage="execute", message="Placing block in pink bowl")
    ...
    server.publish_state(stage="done", message="Ready")
```

If the loop owns the cameras, it also calls `server.publish_frame("side", jpeg_bytes)` for each view.

`boxes` are `{x, y, w, h, score}` as fractions of the full camera frame (0–1). To get them from
PatchCore's anomaly map, take the top 5–8 local peaks of the map and put a box around each. If the
model ran on a crop, add the crop's offset before dividing by the frame size.
