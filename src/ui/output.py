from typing import Optional

import supervisely as sly

from supervisely.app.widgets import (
    Container,
    Card,
    DestinationProject,
    Button,
    ProjectThumbnail,
    Progress,
    Text,
)

import src.globals as g

destination = DestinationProject(
    workspace_id=g.STATE.selected_workspace, project_type="images"
)

start_button = Button("Start button")
stop_button = Button("Stop button", button_type="danger", icon="zmdi zmdi-stop")
stop_button.hide()

progress = Progress()
progress.hide()

result_text = Text()
result_text.hide()

project_thumbnail = ProjectThumbnail()
project_thumbnail.hide()

card = Card(
    title="3️⃣ Output",
    description="PLACEHOLDER: Input description here.",
    content=Container(
        [
            destination,
            start_button,
            progress,
            result_text,
            project_thumbnail,
        ]
    ),
    content_top_right=stop_button,
    lock_message="Save settings on step 2️⃣.",
    collapsable=True,
)
# * Card can be locked and collapsed by default and unlocked and expanded by some conditions in previous steps.
# card.lock()
# card.collapse()


@start_button.click
def generate():
    project_id = destination.get_selected_project_id()
    dataset_id = destination.get_selected_dataset_id()

    result_text.hide()
    project_thumbnail.hide()

    start_button.text = "Working..."
    stop_button.show()

    sly.logger.debug(
        f"Readed values from destination widget. "
        f"Project ID: {project_id}, dataset ID: {dataset_id}."
    )

    if not project_id:
        sly.logger.debug("Project ID is not specified, creating a new project.")
        project_id = create_project(destination.get_project_name())
    if not dataset_id:
        sly.logger.debug("Dataset ID is not specified, creating a new dataset.")
        dataset_id = create_dataset(project_id, destination.get_dataset_name())

    # * ###################################
    # * ###### Add your logic here. #######
    # * ###################################

    progress.show()
    progress_total = 100  # * Input your total here.

    with progress(message="Generating images...", total=progress_total) as pbar:
        for i in range(progress_total):
            if not g.STATE.continue_working:
                # * Checking the global variable to stop working (if stop button was clicked).

                sly.logger.debug("Stop button was clicked, stopping generation.")
                break

            # * ###################################
            # * ###### Add your logic here. #######
            # * ###################################

            pbar.update(1)

    if g.STATE.continue_working:
        # * Input your texts here if the stop button was NOT clicked.
        result_text.text = "Successfully finished."
        result_text.status = "success"

    else:
        # * Input your texts here if the stop button was clicked.

        result_text.text = "Was stopped by user."
        result_text.status = "warning"

    result_text.show()

    # * Showing the widget with result project.
    project_thumbnail.set(g.api.project.get_info_by_id(project_id))
    project_thumbnail.show()

    # * Returning button texts to it's default state.
    start_button.text = "Start button"
    stop_button.hide()
    stop_button.loading = False
    stop_button.text = "Stop button"


@stop_button.click
def stop_generation():
    stop_button.text = "Stopping..."
    stop_button.loading = True
    g.STATE.continue_working = False


def create_project(project_name: Optional[str]) -> int:
    if not project_name:
        # * If the project name wasn't specified by user in widget, it should be generated automatically.
        project_name = "PLACEHOLDER FOR PROJECT NAME"

    project = g.api.project.create(
        g.STATE.selected_workspace, project_name, change_name_if_conflict=True
    )
    return project.id


def create_dataset(project_id: int, dataset_name: Optional[str]) -> int:
    if not dataset_name:
        # * If the dataset name wasn't specified by user in widget, it should be generated automatically.
        dataset_name = "PLACEHOLDER FOR DATASET NAME "

    dataset = g.api.dataset.create(
        project_id, dataset_name, change_name_if_conflict=True
    )
    return dataset.id
