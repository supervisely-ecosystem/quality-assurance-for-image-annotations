from supervisely.app.widgets import (
    Button,
    Card,
    Container,
)

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
