import json
import os

from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    # load_dotenv(os.path.expanduser("~/ninja.env"))

api = sly.Api.from_env()

TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()


STORAGE_DIR = sly.app.get_data_dir()
TF_STATS_DIR = "/stats/"

IMAGES_CACHE = {}
META_CACHE = {}
TF_OLD_CHUNKS = []
BATCH_SIZE = 2
