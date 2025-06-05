import os

from dotenv import load_dotenv
load_dotenv()

""" File containing variables. """
WORK_DIR = r'C:\Users\david\Documents\Projects\rfp-lender-matching-engine'

DATA_PATH = os.path.join(WORK_DIR, "data")
DATA_PATH_RAW = os.path.join(DATA_PATH, "01-raw")
DATA_PATH_WRANGLE = os.path.join(DATA_PATH, "02-cleaned")
DATA_PATH_RESULTS = os.path.join(DATA_PATH, "03-processed")

REGIONAL_HIERARCHY = {
    "usa - west coast": "north america",
    "usa - east coast": "north america",
    "usa - midwest": "north america",
    "usa - south": "north america",
    "western europe": "emea (europe, middle east, africa)",
    "eastern europe": "emea (europe, middle east, africa)",
    "nordics": "emea (europe, middle east, africa)",
    "southeast asia": "apac (asia-pacific)",
    "east asia": "apac (asia-pacific)",
    "south asia": "apac (asia-pacific)",
    "australia/new zealand": "apac (asia-pacific)",
    "middle east & north africa (mena)": "emea (europe, middle east, africa)",
    "sub-saharan africa": "emea (europe, middle east, africa)",
    "canada": "north america",
    "uk & ireland": "emea (europe, middle east, africa)",
    "north america": "global",
    "emea (europe, middle east, africa)": "global",
    "apac (asia-pacific)": "global",
    "latam (latin america)": "global",
    "global": "global"
}