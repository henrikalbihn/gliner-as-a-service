import os

from celery import Celery

gliner_app = Celery(
    "gliner_app",
    broker=os.environ.get("REDIS_URL"),
    backend=os.environ.get("REDIS_URL"),
    include=["app.src.tasks"],
)
