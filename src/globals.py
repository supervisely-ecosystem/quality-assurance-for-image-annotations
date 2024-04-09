import json
import os

from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    # load_dotenv(os.path.expanduser("~/ninja.env"))

api = sly.Api.from_env()

# TEAM_ID = sly.env.team_id()
# WORKSPACE_ID = sly.env.workspace_id()

STORAGE_DIR = sly.app.get_data_dir()
TF_STATS_DIR = "/stats"


PROJ_IMAGES_CACHE = {}
IMAGES_CACHE = {}
META_CACHE = {}
# TF_OLD_CHUNKS = []

CHUNKS_LATEST_DATETIME = None
ACTIVE_REQUESTS_DIR = f"{STORAGE_DIR}/_active_requests"
sly.fs.mkdir(ACTIVE_REQUESTS_DIR, remove_content_if_exists=True)
TF_ACTIVE_REQUESTS_DIR = f"{TF_STATS_DIR}/_active_requests"


def initialize_global_cache():
    global PROJ_IMAGES_CACHE
    PROJ_IMAGES_CACHE = {}
    global IMAGES_CACHE
    IMAGES_CACHE = {}
    global META_CACHE
    META_CACHE = {}
    global TF_OLD_CHUNKS
    TF_OLD_CHUNKS = []


CHUNK_SIZE = 1000
