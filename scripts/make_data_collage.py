"""Build a moving collage of every recorded episode for the site.

289 episodes = 17 x 17, so the collage is a 17 x 17 grid of tiles (top camera by default). Each tile plays its
whole episode, sped up so every tile lasts the same number of frames, and the grid loops.

Run with the anomalib env (it has OpenCV and PyAV):
    C:\\Users\\theoh\\venvs\\anomalib-env\\Scripts\\python.exe scripts/make_data_collage.py

Writes docs/assets/video/collage_289.mp4 and a still, docs/assets/img/collage_289.jpg.
Add "mix" to give each tile one of the three cameras at random instead of always the top camera:
writes collage_289_mix.mp4 / .jpg.
"""

import sys
from pathlib import Path

import av
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets"
MIX = "mix" in sys.argv[1:]
NAME = "collage_289_mix" if MIX else "collage_289"
OUT_VIDEO = REPO / "docs" / "assets" / "video" / f"{NAME}.mp4"
OUT_STILL = REPO / "docs" / "assets" / "img" / f"{NAME}.jpg"

CAMERAS = ["top_view.mp4", "side_view.mp4", "claw_cam.mp4"]
GRID = 17               # 17 x 17 = 289 tiles
TILE_W, TILE_H = 64, 48  # 4:3, like the cameras
FPS = 15
SECONDS = 8
N_FRAMES = FPS * SECONDS


def read_sped_up(video_path):
    """Return N_FRAMES small frames sampled evenly across the whole episode."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA))
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {video_path}")

    picks = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
    return [frames[i] for i in picks]


def main():
    episodes = sorted(DATASETS.glob("*/videos/episode_*"))
    print(f"{len(episodes)} episodes found")
    if len(episodes) != GRID * GRID:
        raise SystemExit(f"expected {GRID * GRID} episodes, found {len(episodes)}")

    # Fixed seed: the same collage every run. Shuffle so neighbouring tiles come from different
    # datasets, and in mix mode pick one camera per tile.
    rng = np.random.default_rng(17)
    order = rng.permutation(len(episodes))
    if MIX:
        videos = [episodes[i] / CAMERAS[rng.integers(3)] for i in range(len(episodes))]
    else:
        videos = [episode / CAMERAS[0] for episode in episodes]

    canvas = np.zeros((N_FRAMES, GRID * TILE_H, GRID * TILE_W, 3), dtype=np.uint8)
    for slot, index in enumerate(order):
        row, col = divmod(slot, GRID)
        tiles = read_sped_up(videos[index])
        y, x = row * TILE_H, col * TILE_W
        for t in range(N_FRAMES):
            canvas[t, y:y + TILE_H, x:x + TILE_W] = tiles[t]
        if (slot + 1) % 50 == 0:
            print(f"  {slot + 1} tiles done")

    # Encode as H.264 so every browser can play it.
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(OUT_VIDEO), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = canvas.shape[2]
        stream.height = canvas.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "28", "preset": "slow", "movflags": "+faststart"}
        for t in range(N_FRAMES):
            rgb = cv2.cvtColor(canvas[t], cv2.COLOR_BGR2RGB)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    # A still from the middle, shown before the video loads.
    OUT_STILL.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_STILL), canvas[N_FRAMES // 2], [cv2.IMWRITE_JPEG_QUALITY, 80])

    size_mb = OUT_VIDEO.stat().st_size / 1e6
    print(f"wrote {OUT_VIDEO.name} ({size_mb:.1f} MB) and {OUT_STILL.name}")


if __name__ == "__main__":
    main()
