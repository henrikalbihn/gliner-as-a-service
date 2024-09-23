"""
API router.
"""

from celery import Task
from celery.result import AsyncResult
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.requests import Request

from .models import PredictRequest, PredictResponse, Task
from .tasks import predict

router = APIRouter()


@router.get("/")
async def root() -> JSONResponse:
    """GLiNER API root endpoint."""
    return JSONResponse(
        status_code=200,
        content={
            "message": "Hello GLiNER",
        },
    )


@router.get("/health")
async def health_endpoint(request: Request) -> JSONResponse:
    """Health check endpoint."""
    logger.debug(f"Request:\n\n{request}")
    is_redis_alive = await request.app.redis.ping()
    return JSONResponse(
        status_code=200,
        content={
            "message": "Ok" if is_redis_alive else "Error",
        },
    )


@router.post("/predict", response_model=Task, status_code=202)
async def predict_endpoint(
    request: Request,
    predict_request: PredictRequest,
) -> JSONResponse:
    """Predict the named entities in the input text."""
    logger.debug(f"Request:\n\n{request}")
    logger.info(f"Object:\n\n{predict_request}")
    task_id = predict.delay(**predict_request.dict())
    return JSONResponse(
        status_code=202,
        content={
            "task_id": str(task_id),
            "status": "Processing",
        },
    )


@router.get(
    "/result/{task_id}",
    response_model=PredictResponse,
    status_code=200,
)
async def result_endpoint(task_id: str) -> JSONResponse:
    """Get the result of the prediction task."""
    logger.info(f"Task ID: {task_id}")
    task = AsyncResult(task_id)
    if not task.ready():
        logger.info("Task is not ready")
        return JSONResponse(
            status_code=202,
            content={
                "task_id": str(task_id),
                "status": "Processing",
            },
        )
    result = task.get()
    logger.info(f"Result: {result}")
    return JSONResponse(
        status_code=200,
        content={
            "task_id": task_id,
            "status": "Success",
            "result": {"predictions": result},
        },
    )
