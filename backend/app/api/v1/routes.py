from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import shutil
import os

from app.settings import settings
from app.pipeline import process_video as run_pipeline

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[3]


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


@router.post("/process")
async def process_video(background_tasks: BackgroundTasks, video: UploadFile = File(...), fast: bool = False):
    if not video:
        raise HTTPException(status_code=400, detail="No video uploaded")
    if not video.content_type or not video.content_type.startswith("video"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")

    # Persist upload and outputs in temp files
    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_f:
        in_path = in_f.name
        await video.seek(0)
        shutil.copyfileobj(video.file, in_f)

    inter_fd, inter_path = tempfile.mkstemp(suffix=".mp4")
    os.close(inter_fd)
    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)

    # Run the processing pipeline in-process
    try:
        run_pipeline(
            input_path=Path(in_path),
            output_path=Path(out_path),
            model_path=Path(settings.model_path),
            intermediate_path=Path(inter_path),
            fast=fast,
        )
    except Exception as e:
        _cleanup([in_path, inter_path, out_path])
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        _cleanup([in_path, inter_path, out_path])
        raise HTTPException(status_code=500, detail="Processing did not produce an output video (empty file)")

    background_tasks.add_task(_cleanup, [in_path, inter_path, out_path])
    return FileResponse(out_path, media_type="video/mp4", filename="analyzed.mp4")
