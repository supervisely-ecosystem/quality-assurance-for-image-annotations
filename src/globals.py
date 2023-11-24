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
PROJECT_ID = sly.env.project_id()
# DATASET_ID = sly.env.dataset_id(raise_not_found=False)

STORAGE_DIR = sly.app.get_data_dir()
DST_TF_DIR = "/stats/"
