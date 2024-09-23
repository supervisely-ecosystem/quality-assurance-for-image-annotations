FROM supervisely/base-py-sdk:6.73.137

# COPY dev_requirements.txt dev_requirements.txt
# RUN python3 -m pip install -r dev_requirements.txt
# git+https://github.com/supervisely/supervisely.git@figures_download_all_pages

RUN pip install dataset-tools==0.1.3
RUN pip install humanize
RUN pip install ujson
RUN pip install scikit-image==0.21.0
RUN pip install Pillow==9.5.0

LABEL python_sdk_version=6.73.137