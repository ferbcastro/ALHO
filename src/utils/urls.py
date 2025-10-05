from urllib.parse import urlparse
import string
import re

CHAR_SPACE = string.printable[:-6] # printable characters except whitespaces
CHAR_SPACE_LEN = len(CHAR_SPACE)
CHAR_INDEX = {c: i for i, c in enumerate(CHAR_SPACE)}

# Strip scheme and characters outside CHAR_SPACE
def strip_url(url: str) -> str:
    url = "".join(char for char in url if char in CHAR_SPACE)

    if (scheme := urlparse(url).scheme):
        url = re.sub(f"^{scheme}://", "", url)

    return url