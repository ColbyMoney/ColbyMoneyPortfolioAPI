"""Root FastAPI application — mounts all project routers."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_four_in_a_row.router import router as ai_four_in_a_row_router

app = FastAPI(title="Colby Money API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://colbymoney.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_four_in_a_row_router, prefix="/api/ai-four-in-a-row")
