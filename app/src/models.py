"""
Pydantic models for the API.
"""

from textwrap import dedent

from pydantic import BaseModel


class Task(BaseModel):
    """Celery task representation"""

    task_id: str
    status: str


class PredictResponse(BaseModel):
    """Predict Response"""

    task_id: str
    status: str
    result: dict


class PredictRequest(BaseModel):
    """Request body"""

    inputs: list[str] = [
        dedent(
            """
        This is a story all about how my life got flipped turned upside down and I'd like to take a minute just sit right there I'll tell you how I became the prince of a town called Bel-Air.
        """
        ).strip(),
    ]
    labels: list[str] = [
        "PERSON",
        "PLACE",
        "THING",
        "ORGANIZATION",
        "DATE",
        "TIME",
    ]  # This is just a default, can be anything the user wants
    flat_ner: bool = True
    threshold: float = 0.3
    multi_label: bool = False
    batch_size: int = 12
