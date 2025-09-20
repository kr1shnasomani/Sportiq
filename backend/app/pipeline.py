"""Pipeline adapter for the FastAPI endpoint.

We invoke the legacy script `code/court_detection.py` as a subprocess to produce
the intermediate and final output videos. This keeps heavy OpenCV/Torch work
isolated from the web server process and avoids import path issues.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def process_video(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    intermediate_path: Path,
    fast: bool = False,
) -> None:
    """Run the legacy pipeline script to process a video.

    Args:
        input_path: Path to the input video.
        output_path: Path to write the final analyzed video (mp4).
        model_path: Path to the TrackNet model .pth file.
        intermediate_path: Path for the warped-court intermediate output.
        fast: Currently unused; reserved for future flags.
    """

    backend_root = Path(__file__).resolve().parents[1]
    # Use the snake_case filename present in backend/code
    script_path = backend_root / "code" / "court_detection.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_path),
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--intermediate-output",
        str(intermediate_path),
    ]

    # Launch with cwd set to backend root so relative paths inside the script work
    proc = subprocess.run(
        cmd,
        cwd=str(backend_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed (exit={proc.returncode}). Output:\n{proc.stdout}"
        )