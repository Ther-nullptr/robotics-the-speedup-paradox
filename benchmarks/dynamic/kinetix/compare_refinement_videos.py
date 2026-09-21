"""Compose matched simulator MP4s without changing the recorded trajectories."""

import argparse
import json
import os
from pathlib import Path


def compose(run_dir, task, variants, fps):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The simulation run must finish before video composition")
    rows = [r for r in manifest["rows"] if r["task"] == task]
    reference = next(r for r in rows if r["factor"] == 1)
    selected = [reference]
    for name in variants:
        matches = [r for r in rows if r["factor"] > 1 and r["variant"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one refined row for {task}/{name}")
        selected.append(matches[0])
    tapes = {r["action_tape_sha256"] for r in selected}
    if len(tapes) != 1:
        raise ValueError("Videos must use one identical action tape")
    common_control = min(r["observed_controls"] for r in selected)
    trajectories = []
    for row in selected:
        with np.load(run_dir / f"{row['name']}.npz", allow_pickle=False) as data:
            trajectories.append(data["position"])
    if not all(np.array_equal(x[0], trajectories[0][0]) for x in trajectories):
        raise ValueError("Videos must use the same initial body positions")
    errors = [
        float(
            np.sqrt(np.mean((x[common_control] - trajectories[0][common_control]) ** 2))
        )
        for x in trajectories
    ]
    common_seconds = common_control * reference["control_dt_seconds"]
    videos = []
    for row in selected:
        with imageio.get_reader(run_dir / row["video"]) as reader:
            # ffmpeg readers can report an unbounded length hint to list().
            frames = [frame for frame in reader]
        if len(frames) != row["observed_controls"] + 1:
            raise ValueError("Video frame count does not match the observed controls")
        videos.append(frames)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
        small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()
    width, height, header, footer = 500, 500, 126, 104
    count = max(map(len, videos))
    preview = []
    output = run_dir / f"{task}_comparison.mp4"
    with imageio.get_writer(output, fps=fps, macro_block_size=1) as writer:
        for index in range(count + fps * 2):
            canvas = Image.new(
                "RGB", (width * len(selected), header + height + footer), "white"
            )
            draw = ImageDraw.Draw(canvas)
            for column, (row, frames) in enumerate(zip(selected, videos, strict=True)):
                current = min(index, len(frames) - 1)
                x = column * width
                draw.rectangle((x, 0, x + width, header), fill="#ecf0f5")
                title = (
                    "Original"
                    if row["factor"] == 1
                    else row["variant"].replace("_", " ")
                )
                draw.text((x + 12, 10), title, font=font, fill="#122337")
                lines = [
                    f"Physics dt: {row['physics_dt_seconds'] * 1000:.6g} ms",
                    f"Motor feedback: {row['motor_feedback_hz']:g} Hz",
                    f"Sim time: {current * row['control_dt_seconds']:.3f} s | control {current}",
                ]
                for line, label in enumerate(lines):
                    draw.text(
                        (x + 12, 39 + line * 25), label, font=small, fill="#122337"
                    )
                canvas.paste(Image.fromarray(frames[current]), (x, header))
                ended = index >= len(frames) - 1
                if ended:
                    limit = (
                        "OBSERVATION LIMIT"
                        if row["termination_reason"] == "observation_limit"
                        else "TAPE END"
                    )
                    status = (
                        f"{limit}: no terminal observed"
                        if row["censored"]
                        else "SUCCESS"
                        if row["success"]
                        else "TERMINATED: failure"
                    )
                    draw.rectangle(
                        (x, header + height - 36, x + width, header + height),
                        fill="#122337",
                    )
                    draw.text(
                        (x + 12, header + height - 27),
                        status + " [held]",
                        font=small,
                        fill="white",
                    )
                rmse = errors[column]
                draw.text(
                    (x + 12, header + height + 10),
                    f"Position RMSE at t={common_seconds:.3f} s: {rmse:.6g}",
                    font=small,
                    fill="#122337",
                )
            y = header + height
            draw.text(
                (12, y + 42),
                f"{task} | Same initial state and command tape | Actual physics integration in every substep",
                font=small,
                fill="#122337",
            )
            draw.text(
                (12, y + 70),
                f"Reward sampled at native 60 Hz times | Playback {fps / 30:.3g}x | 30 Hz recorded states; no trajectory interpolation",
                font=small,
                fill="#122337",
            )
            writer.append_data(np.asarray(canvas))
            if index in {0, count // 2, count - 1}:
                preview.append(canvas)
    contact = Image.new(
        "RGB",
        (width * len(selected), (header + height + footer) * len(preview)),
        "white",
    )
    for index, frame in enumerate(preview):
        contact.paste(frame, (0, frame.height * index))
    contact.save(run_dir / f"{task}_comparison_preview.png")
    return {
        "task": task,
        "video": output.name,
        "frames": count + fps * 2,
        "fps": fps,
        "variants": [r["name"] for r in selected],
        "common_control": common_control,
        "position_rmse_at_common_time": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variants", default="direct,held_motor_collision")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    os.environ["IMAGEIO_FFMPEG_NO_PREVENT_SIGINT"] = "1"
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    reports = [
        compose(args.run_dir, task, args.variants.split(","), args.fps)
        for task in sorted({r["task"] for r in manifest["rows"]})
    ]
    (args.run_dir / "comparison_videos.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
