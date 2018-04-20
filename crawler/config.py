

# information about wikiart
BASE_URL= "https://www.wikiart.org/en/"
STYLE_URL_PREFIX = "paintings-by-style/"
PAGINATION_URL_PREFIX = "?json=2&page="


# kind of styles we want
STYLES = ["gongbi", "gothic", "northern-renaissance", 
          "realism", "ink-and-wash-painting","abstract-art", 
          "shin-hanga", "international-gothic", "impressionism"]

# timeout for WikiArt
# SEE https://github.com/lucasdavid/wikiart/blob/master/wikiart/settings.py
METADATA_REQUEST_TIMEOUT = 2 * 60
PAINTINGS_REQUEST_TIMEOUT = 5 * 60


STYLE_JSON = "../data/json/styles.json"
JSON_PATH = "../data/json"

IMAGE_PATH = "../data/paintings"


PAGE_LIMIT = 20

SAVE_IMAGES_IN_FORMAT = '.jpg'