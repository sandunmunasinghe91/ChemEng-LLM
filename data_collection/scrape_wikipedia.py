
import wikipedia
import time
import logging
import os
from urllib.request import urlretrieve
import requests
import json
import re
import wikipediaapi
from src.utils.retry import make_request

API_URL = 'http://en.wikipedia.org/w/api.php'
MAX_RESULTS = 50
RATE_LIMIT = False
RATE_LIMIT_MIN_WAIT = None
RATE_LIMIT_LAST_CALL = None
USER_AGENT = 'wikipedia (https://github.com/goldsmith/Wikipedia/)'

DOWNLOAD_DIR = 'data/raw/wikipedia'
os.makedirs(name=DOWNLOAD_DIR,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Chem-LLM/1.0 (chemical engineering research project)"
}
wiki = wikipediaapi.AsyncWikipedia(user_agent='Chem-LLM ', language='en')
print("Current working directory:", os.getcwd())
print("Wikipedia download directory:", os.path.abspath(DOWNLOAD_DIR))

queries = [
    # 1. Chemical engineering fundamentals
    "chemical engineering thermodynamics",
    "transport phenomena chemical engineering",
    "fluid mechanics chemical engineering",
    "heat transfer chemical engineering",
    "mass transfer chemical engineering",

    # 2. Reaction engineering
    "chemical reaction engineering",
    "chemical reaction kinetics",
    "chemical reactor modeling",
    "chemical reactor design",
    "CSTR reactor",
    "plug flow reactor",
    "catalytic reactor",

    # 3. Separation processes
    "chemical separation processes",
    "distillation process",
    "vapor liquid equilibrium",
    "absorption process chemical engineering",
    "adsorption process chemical engineering",
    "membrane separation",
    "liquid liquid extraction",

    # 4. Process engineering
    "process systems engineering",
    "chemical process design",
    "chemical process simulation",
    "chemical process optimization",
    "process integration chemical engineering",
    "process intensification",

    # 5. Process dynamics and control
    "chemical process control",
    "process dynamics chemical engineering",
    "model predictive control chemical process",
    "PID control chemical process",
    "advanced process control",
    "process control optimization",

    # 6. Plant monitoring
    "chemical process monitoring",
    "industrial process monitoring",
    "process fault detection",
    "process fault diagnosis",
    "chemical process anomaly detection",
    "sensor fault detection industrial process",

    # 7. Equipment
    "heat exchanger chemical engineering",
    "distillation column modeling",
    "chemical reactor operation",
    "industrial pumps process engineering",
    "compressor process engineering",
    "industrial boiler process",
    "pressure vessel chemical engineering",

    # 8. Safety
    "chemical process safety",
    "process hazard analysis",
    "chemical plant risk assessment",
    "industrial process safety",
    "chemical process accident prevention",

    # 9. Troubleshooting / predictive maintenance
    "chemical process troubleshooting",
    "industrial process fault diagnosis",
    "predictive maintenance chemical plant",
    "predictive maintenance process industry",
    "equipment failure detection industrial process",
    "remaining useful life industrial equipment",

    # 10. AI / data-driven process engineering
    "machine learning chemical engineering",
    "machine learning chemical process",
    "deep learning process engineering",
    "artificial intelligence chemical process",
    "digital twin chemical process",
    "digital twin chemical plant",
    "industrial process optimization machine learning",
]

def search_wikipedia(query):
    'Search the Wikipedia using the MediaWiki API and return the search results.'
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": MAX_RESULTS,
        "format": "json",
    }
  
    data = make_request(
    url=API_URL,
    params=params,
    headers=HEADERS,
    )
    return data["query"]["search"]

def get_article(page_id):
    "Retrieve the article text and page information using its page ID."
    params = {
        "action": "query",
        "prop": "extracts|info",
        "pageids": page_id,
        "explaintext": True,
        "inprop": "url",
        "format": "json",
    }

    data = make_request(
    url=API_URL,
    params=params,
    headers=HEADERS,
    )
    
    return data["query"]["pages"][str(page_id)]


def clean_filename(title):

    # Remove characters that can cause filename problems
    title = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        title
    )

    # Replace spaces with underscores
    title = title.replace(" ", "_")

    return title


downloaded = 0
skipped = 0
failed = 0

seen_page_ids = set()

for query in queries:
    logger.info(f"Searching: {query}")
    try:
        search_results = search_wikipedia(query)
    except Exception as e:
        logger.error(
            f"Search failed for '{query}': {e}"
        )
        failed += 1
        continue

    for result in search_results:
        page_id = result["pageid"]
        title = result["title"]

        if page_id in seen_page_ids:
            skipped += 1
            logger.info(
                f"Duplicate, skipping: {title}"
            )
            continue

        try:
            page = get_article(page_id)
            text = page.get("extract", "")

            # Skip pages without useful text
            if not text.strip():
                skipped += 1
                logger.warning(
                    f"No text found, skipping: {title}"
                )
                continue

            article = {
                "page_id": page_id,
                "title": page["title"],
                "url": page.get("fullurl"),
                "source": "Wikipedia",
                "search_query": query,
                "text": text
            }

            file_name = (
                f"{page_id}_"
                f"{clean_filename(page['title'])}.json"
            )
            file_path = os.path.join( DOWNLOAD_DIR, file_name)

            if os.path.exists(file_path):
                skipped += 1
                seen_page_ids.add(page_id)
                logger.info(
                    f"Already exists, skipping: {file_name}"
                )
                continue

            with open(file_path, "w",encoding="utf-8" ) as f:

                json.dump(article, f, ensure_ascii=False, indent=2)

            seen_page_ids.add(page_id)
            downloaded += 1
            logger.info(
                f"Downloaded: {downloaded} | "
                f"{page['title']}"
            )

            time.sleep(0.5)

        except Exception as e:
            failed += 1
            logger.error(
                f"Failed: {title}: {e}"
            )

logger.info(
    f"Finished | "
    f"Downloaded: {downloaded} | "
    f"Skipped: {skipped} | "
    f"Failed: {failed}"
)
