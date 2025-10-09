import os

_PATH_PROJECT_ROOT = os.path.dirname(__file__)
PATH_RESOURCES = os.path.join(_PATH_PROJECT_ROOT, "resources")
PATH_SFCR_REPORTS = os.path.join(PATH_RESOURCES, "sfcr")
PATH_SCHEMA = os.path.join(_PATH_PROJECT_ROOT, "schema")
FOLDER_NAME_PKV_REPORTS = "pkv"
PATH_SFCR_REPORTS_PKV = os.path.join(PATH_SFCR_REPORTS, FOLDER_NAME_PKV_REPORTS)

PATH_OUTPUT = os.path.join(_PATH_PROJECT_ROOT, "output")
