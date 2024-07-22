import os
from dotenv import load_dotenv
import supervisely as sly
from supervisely.sly_logger import LOGGING_LEVELS


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    # load_dotenv(os.path.expanduser("~/ninja.env"))

api = sly.Api.from_env()

STORAGE_DIR = sly.app.get_data_dir()
TF_STATS_DIR = "/stats"


CHUNKS_LATEST_DATETIME = None
ACTIVE_REQUESTS_DIR = f"{STORAGE_DIR}/_active_requests"
sly.fs.mkdir(ACTIVE_REQUESTS_DIR, remove_content_if_exists=True)
TF_ACTIVE_REQUESTS_DIR = f"{TF_STATS_DIR}/_active_requests"

CHUNK_SIZE: int = 1000
MINIMUM_DTOOLS_VERSION: str = (
    "0.0.92"  # force stats to fully recalculate (f.e. when edit statistics)
)
HEALTHCHECK_PROJECT_ID = 10387


def _initialize_log_levels(project_id):
    global _INFO
    global _DEBUG
    global _WARNING
    _INFO = LOGGING_LEVELS["INFO"].int
    _DEBUG = LOGGING_LEVELS["DEBUG"].int
    _WARNING = LOGGING_LEVELS["WARN"].int
    if project_id == HEALTHCHECK_PROJECT_ID:
        _INFO = LOGGING_LEVELS["DEBUG"].int
        _WARNING = LOGGING_LEVELS["DEBUG"].int
