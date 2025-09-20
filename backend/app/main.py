from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.settings import settings
from app.api.v1.routes import router as v1_router

app = FastAPI(title="Sportiq Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(v1_router, prefix="/api")


@app.get("/")
async def root():
    return JSONResponse({
        "message": "Sportiq Backend is running",
        "endpoints": ["GET /health", "POST /api/process (multipart form field: video)"]
    })
