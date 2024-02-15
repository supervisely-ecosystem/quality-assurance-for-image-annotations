import json
import os
import time
from tqdm import tqdm
import cv2
import numpy as np

import src.globals as g
import src.utils as u
import supervisely as sly
from supervisely.io.fs import get_file_name_with_ext
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Editor,
    Empty,
    Image,
    SelectItem,
    Text,
    SelectProject,
)

import dataset_tools as dtools

button_stats = Button(text="Calculate")
# button_save = Button(text="Save settings")
# infotext = Text("Settings saved", "success")
# select_item = SelectItem(dataset_id=None, compact=False)

# select_item = SelectProject(g.PROJECT_ID, g.WORKSPACE_ID)


card_1 = Card(
    title="Calculate stats",
    content=Container(
        widgets=[
            # select_item,
            button_stats,
        ]
    ),
)

# infotext.hide()
