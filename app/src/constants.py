import os

VALKEY_URL = os.environ.get("VALKEY_URL", "valkey://valkey:6379/0")
REDIS_URL = os.environ.get("REDIS_URL", VALKEY_URL)
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "GLiNER-S")
