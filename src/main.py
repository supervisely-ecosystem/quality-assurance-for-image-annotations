from pathlib import Path

import cv2
from fastapi import Response

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui.input import card_1
from supervisely.app.widgets import Container

layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
server = app.get_server()
