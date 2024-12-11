from celery import Celery

from .constants import REDIS_URL

# Celery doesn't support valkey://
# transport protocol, so we use redis://
# as a workaround
gliner_app = Celery(
    "gliner_app",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.src.tasks"],
)
