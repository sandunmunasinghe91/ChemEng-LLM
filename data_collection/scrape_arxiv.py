import arxiv
import time
import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

DOWNLOAD_DIR = Path('data/raw/arxiv_papers')
os.makedirs(name=DOWNLOAD_DIR,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

client = arxiv.Client()
logger.info('arXiv client initialized')

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


downloaded = 0
skipped = 0
failed = 0
number = 6200

for query in queries:
    search = arxiv.Search(
        query= query,
        max_results=100
    )
    result = client.results(search)
    for paper in result:
        paper_id = paper.get_short_id()
        file_name = f'{paper_id}.pdf'
        file_path = os.path.join(DOWNLOAD_DIR,file_name)

        if os.path.exists(file_path):
            skipped += 1
            logger.info(f'Already exists: skipping: {file_name}')
            continue 

        try:
            urlretrieve(paper.pdf_url,file_path)
            downloaded += 1
            logger.info(f'downloaded: {file_name}')
        except Exception as e:
            failed += 1
            logger.error(f'Failed to download: {file_name}: {e}')
    time.sleep(3)


logger.info(
    f" Downloaded:{downloaded}/{number} | "
    f" Skipped: {skipped} | Failed: {failed}"
)