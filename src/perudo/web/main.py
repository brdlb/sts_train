"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import web_config
from .database.database import init_db
from .game_server import GameServer
from .api import games, models, statistics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    print("Initializing database...")
    init_db()

    print("Initializing game server...")
    game_server = GameServer()

    # Set game server in API modules
    games.set_game_server(game_server)
    models.set_game_server(game_server)

    # Store in app state
    app.state.game_server = game_server

    yield

    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Perudo Web API",
    description="API for playing Perudo game with AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=web_config.cors_origins,
    allow_origin_regex=web_config.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(games.router)
app.include_router(models.router)
app.include_router(statistics.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Perudo Web API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.perudo.web.main:app",
        host=web_config.host,
        port=web_config.port,
        reload=web_config.debug,
    )

