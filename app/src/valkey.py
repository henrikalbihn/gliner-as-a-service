"""
Valkey utilities.
"""

from contextlib import asynccontextmanager

import valkey.asyncio as valkey
from fastapi import FastAPI
from loguru import logger

from .constants import VALKEY_URL


async def get_valkey() -> valkey.Valkey:
    """Get the valkey connection."""
    pool = valkey.ConnectionPool.from_url(
        VALKEY_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    client = valkey.Valkey.from_pool(pool)
    test = await client.ping()
    logger.info(f"Valkey connection test: {test}")
    return client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application."""
    try:
        # Load the valkey connection
        app.valkey = await get_valkey()
        yield
    finally:
        # close valkey connection and release the resources
        await app.valkey.aclose(close_connection_pool=True)
