import os
from dotenv import load_dotenv
import supervisely as sly

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
    "0.0.68"  # force stats to fully recalculate (f.e. when added new statistics)
)
