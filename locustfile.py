"""
GLiNER Locust load test script.
"""

import random
import time
from textwrap import dedent

from datasets import load_dataset
from locust import HttpUser, between, task
from loguru import logger

PORT = 8080
HOST = f"http://fastapi:{PORT}"

# https://huggingface.co/datasets/gsm8k
DATASET_ID = "gsm8k"


def load_data() -> dict:
    logger.info("Loading dataset...")
    DATA = load_dataset(DATASET_ID, "main")
    logger.info("Dataset loaded.")
    return DATA


def get_random_sample() -> str:
    idx = random.randint(0, len(DATA["test"]))
    sample = DATA["test"][idx]
    return dedent(
        f"""
        {sample['question']}

        {sample['answer']}
        """
    )


DATA = load_data()


class GLiNERLoadTest(HttpUser):
    """Locust load test for GLiNER API."""

    host = HOST
    wait_time = between(1, 5)

    def get_named_entities(self) -> None:
        """Send a chat completion request to the GLiNER API."""
        endpoint = "/predict"
        headers = {"Content-Type": "application/json"}
        request_count = 0
        payload = {
            "inputs": [get_random_sample()],
            "labels": [
                "PERSON",
                "PLACE",
                "THING",
                "ORGANIZATION",
                "DATE",
                "TIME",
            ],
        }
        resp = self.client.post(endpoint, headers=headers, json=payload)
        if resp.status_code not in (200, 202):
            logger.error(f"Request failed: {resp.text}")
            return
        request_count += 1
        resp = resp.json()
        task_id = resp.get("task_id", None)
        if not task_id:
            return
        time.sleep(2)
        endpoint = f"/result/{task_id}"
        resp = self.client.get(endpoint, headers=headers)
        request_count += 1
        resp = resp.json()
        status = resp.get("status", None)
        while status == "Processing":
            time.sleep(5)
            resp = self.client.get(endpoint, headers=headers)
            request_count += 1
            resp = resp.json()
            status = resp.get("status", None)
            if status == "Success":
                logger.info("Task completed.")
                break
        logger.info(f"Request count: {request_count}")

    @task
    def execute_task(self) -> None:
        """Execute tasks."""
        self.get_named_entities()
