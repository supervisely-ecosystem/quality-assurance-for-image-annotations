from supervisely.app.widgets import Text, Card

example_text = Text("This is a widget for settings.", "info")


card = Card(
    title="2️⃣ Settings",
    description="PLACEHOLDER: Input description here.",
    content=example_text,
    lock_message="Select the dataset on step 1️⃣.",
    collapsable=True,
)
card.lock()
card.collapse()
