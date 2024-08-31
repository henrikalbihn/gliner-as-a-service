"""
Celery tasks for the Gliner service.
"""

from typing import Any

from celery import Task
from loguru import logger

from app.src.celery_worker import gliner_app
from app.src.gliner import NERModel


class PredictTask(Task):
    """Predict Task"""

    abstract = True

    def __init__(self) -> None:
        """Init predict task."""
        super().__init__()
        logger.debug("Init predict task.")
        self.model: NERModel = None

    def __call__(self, *args, **kwargs) -> Any:
        """Call predict task."""
        if not self.model:
            # Protecting this in the __call__ method to avoid
            # loading the model in the fastapi server process.
            self.model = NERModel()
        return self.run(*args, **kwargs)


@gliner_app.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
    name="gliner.predict",
)
def predict(
    self,
    inputs: list[str],
    labels: list[str],
    flat_ner: bool = True,
    threshold: float = 0.3,
    multi_label: bool = False,
    batch_size: int = 12,
) -> list[list[dict[str, Any]]]:
    """Predict task.

    Args:
        self (Task): Task instance.
        inputs (list[str]): List of inputs.
        labels (list[str]): List of labels.
        flat_ner (bool, optional): Flat NER. Defaults to True.
        threshold (float, optional): Threshold. Defaults to 0.3.
        multi_label (bool, optional): Multi label. Defaults to False.
        batch_size (int, optional): Batch size. Defaults to 12.

    Returns:
        list[list[dict[str, Any]]]: List of results.
    """
    try:
        logger.info(f"Predicting [{len(inputs):,}] inputs.")
        results = self.model.batch_predict(
            targets=inputs,
            labels=labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
        )
        logger.info("Prediction complete.")
        return results
    except Exception as e:
        logger.exception("Prediction failed.", e)
        raise e
