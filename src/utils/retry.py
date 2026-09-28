

import time
import logging
import requests

logger = logging.getLogger(__name__)

# errors worth retrying — temporary server-side problems
RETRYABLE_STATUS_CODES = {
    429,  # too many requests
    500,  # internal server error
    502,  # bad gateway
    503,  # service unavailable
    504,  # gateway timeout
}

def make_request(
    url: str,
    params: dict = None,
    headers: dict = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    timeout: int = 30,
    return_json:bool = True,
):
    """
    Make a GET request with exponential backoff retry.

    Retries on 429 (rate limit) and common server errors.
    Raises immediately on 4xx client errors (except 429).
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            # handle retryable status codes
            if response.status_code in RETRYABLE_STATUS_CODES:

                # check if API told us how long to wait
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait_time = float(retry_after)
                else:
                    wait_time = min(
                        base_delay * (backoff_factor ** attempt),
                        max_delay,
                    )

                logger.warning(
                    f"HTTP {response.status_code} on attempt "
                    f"{attempt + 1}/{max_retries}. "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
                continue

            # raise immediately on non-retryable errors (400, 401, 403, 404)
            response.raise_for_status()

            # success — polite delay then return
            time.sleep(1.0)
            if return_json:
                return response.json()
            return response

        except requests.exceptions.Timeout as e:
            last_exception = e
            wait_time = min(
                base_delay * (backoff_factor ** attempt),
                max_delay,
            )
            logger.warning(
                f"Request timed out (attempt {attempt + 1}/{max_retries}). "
                f"Waiting {wait_time:.1f}s..."
            )
            time.sleep(wait_time)

        except requests.exceptions.ConnectionError as e:
            last_exception = e
            wait_time = min(
                base_delay * (backoff_factor ** attempt),
                max_delay,
            )
            logger.warning(
                f"Connection error (attempt {attempt + 1}/{max_retries}): "
                f"{e}. Waiting {wait_time:.1f}s..."
            )
            time.sleep(wait_time)

    # all retries exhausted
    raise Exception(
        f"All {max_retries} attempts failed for {url} "
        f"with params {params}. Last error: {last_exception}"
    )