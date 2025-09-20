<h1 align="center">Sportiq</h1>
<p align="center">
  <img src="https://github.com/user-attachments/assets/34e9b4e7-a768-4143-b819-fc726dbf96d3" height="250cm"/>
</p>
Sportiq performs player and ball tracking for tennis. It detects court boundaries, tracks players, and maps ball trajectories using Computer Vision (OpenCV), MediaPipe, and a TrackNet-based model.

## Execution Guide:

The project has a FastAPI backend and a Vite React frontend. Use the root runner to start both in dev mode.

1) Clone the repository and navigate into it:

```bash
git clone https://github.com/kr1shnasomani/Sportiq.git
cd Sportiq
```

2) Start both servers:

```bash
./run_all.sh
```

What this does:
- Creates/uses a local Python virtual environment under `backend/.venv` and installs Python deps from `backend/requirements.txt`.
- Starts the FastAPI backend (auto-picks a free port like 8000/8001 and prints it, health at `/health`).
- Starts the Vite dev server for the frontend (auto-picks port like 8080/8081 and prints it, proxies `/api` to the backend).

Open the printed frontend URL in your browser, upload a tennis video, and download the processed result.

Note:
- The TrackNet weight file lives at `backend/model/TrackNet.pth` (already included here).
- Temporary files are handled automatically; you do not need to modify paths in code.

## Run Services Individually (Optional):

Backend only:

```bash
cd backend
chmod +x run_backend.sh
PORT=8000 ./run_backend.sh
```

Frontend only:

```bash
cd frontend
chmod +x run_frontend.sh
VITE_BACKEND_URL="http://localhost:8000" VITE_DEV_PORT=8080 ./run_frontend.sh
```

## API Usage (Direct):

The backend exposes a single main endpoint to process a video upload.

- Health check: `GET /health`
- Process video: `POST /api/process` (multipart/form-data, file field: `video`)

Example curl:

```bash
curl -X POST \
  -F "video=@/absolute/path/to/input.mp4" \
  http://localhost:8000/api/process \
  -o output.mp4
```

## Output:

https://github.com/user-attachments/assets/31d239fb-44a8-4906-a2e6-43886348ba5b

## Workflow:

![image](https://github.com/user-attachments/assets/3a72bb94-7b6e-46e7-baf8-770a92bd4780)