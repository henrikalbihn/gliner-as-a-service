"""
Redis utilities.
"""

import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from loguru import logger

REDIS_URL = os.getenv("REDIS_URL")


async def get_redis() -> aioredis.Redis:
    """Get the redis connection."""
    pool = aioredis.ConnectionPool.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    client = aioredis.Redis.from_pool(pool)
    test = await client.ping()
    logger.info(f"Redis connection test: {test}")
    return client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application."""
    try:
        # Load the redis connection
        app.redis = await get_redis()
        yield
    finally:
        # close redis connection and release the resources
        await app.redis.aclose(close_connection_pool=True)
