from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import mamba_router

app = FastAPI(title="Susanoo Systems Hybrid Architecture API")

# Add CORS middleware so the local HTML file can query it without Cross-Origin blocking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local prototype
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mamba_router)
