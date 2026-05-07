FROM supervisely/base-py-sdk:6.73.486

# COPY dev_requirements.txt dev_requirements.txt
# RUN python3 -m pip install -r dev_requirements.txt
# git+https://github.com/supervisely/supervisely.git@figures_download_all_pages

RUN python3 -m pip install --no-cache-dir --force-reinstall "git+https://github.com/supervisely/dataset-tools.git@49a092c14852f6edc2f04fc75f0e307e1ffa7375#egg=dataset-tools"
RUN pip install humanize
RUN pip install ujson
RUN pip install scikit-image==0.21.0
RUN pip install Pillow==9.5.0

LABEL python_sdk_version=6.73.486
