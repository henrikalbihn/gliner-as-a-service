"""
Main module to define the FastAPI application.
"""

from textwrap import dedent

from fastapi import FastAPI

from app.src import __app_name__, __version__
from app.src.redis import lifespan
from app.src.router import router

app = FastAPI(
    title=__app_name__,
    version=__version__,
    description=dedent(
        """
        **GLiNER** is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.
        """
    ).strip(),
    summary="Named Entity Recognition API",
    lifespan=lifespan,
)

app.include_router(router)
