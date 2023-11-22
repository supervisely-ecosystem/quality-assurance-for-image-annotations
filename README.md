<div align="center" markdown>
<img src="poster placeholder"/>

# Placeholder for app short description

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/PLACEHOLDER-FOR-APP-PATH)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/PLACEHOLDER-FOR-APP-PATH)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/PLACEHOLDER-FOR-APP-PATH.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/PLACEHOLDER-FOR-APP-PATH.png)](https://supervise.ly)

</div>

## Overview

Section for app overview. Describe what the app does, what are the benefits of using it, what is the expected output, etc.

## Preparation

Section for app preparation. Describe what the user should do before running the app. For example, upload images to the team storage, create a project, etc.

## How To Run

Section for the app running. Describe how to run the app step by step.

**Step 1:** Describe actions in step.<br><br>

**Step 2:** Describe actions in step.<br><br>
<img src="placeholder for screenshot"/><br><br>

After finishing using the app, don't forget to stop the app session manually in the App Sessions. The app will write information about the text prompt and CLIP score to the image metadata. You can find this information in the Image Properties - Info section of the image in the labeling tool.

---

_This section is about how to use the repo template, it should be removed from README.md after the app is ready to be released._<br>

### How to use the repo template

1. Clone the repo locally.
2. Use the `create_venv.sh` script to create a virtual environment and install all required packages.
3. Fill required fields in the `local.env` file (your IDs: team, workspace, etc.).
4. Add the required code to the `src/globals.py` file.
5. The template contains `src/ui/input.py` module, which has ready-to-work code if the app uses images projects or datasets. If you don't need it, you can replace it with your own code.
6. The template also contains `src/ui/output.py` module, which has ready-to-work code for output, but it can be replaced with your own code.
7. After implementing all required code and UI, fill in `config.json` file. If needed edit `requirements.txt` file.
8. The repo is ready now.
