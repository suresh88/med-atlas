"""
Configuration constants for FDA data extraction.

This module centralizes configurable parameters for the FDA
drug approval extractor. Adjust these values as needed
without modifying the main extractor class.

Attributes
==========
FDA_API_BASE_URL : str
    Base URL for the openFDA Drugs@FDA search endpoint.
RATE_LIMIT_DELAY : float
    Seconds to wait between successive API requests.  Adjust to
    respect rate limits and avoid throttling by the API service.
MAX_RETRIES : int
    Maximum number of retries for each API call when facing
    network errors or HTTP 5xx responses.
TIMEOUT_SECONDS : int
    HTTP timeout in seconds for each API call.
BATCH_SIZE : int
    Number of records to process in each batch when working with
    large datasets.  Adjust according to available memory and
    desired progress granularity.
"""

# Base endpoint for the openFDA Drugs@FDA search API
FDA_API_BASE_URL: str = "https://api.fda.gov/drug/drugsfda.json"

# Delay (seconds) between API requests to respect rate limits
RATE_LIMIT_DELAY: float = 1.0

# Maximum number of retries for failed API calls
MAX_RETRIES: int = 3

# Request timeout (seconds)
TIMEOUT_SECONDS: int = 30

# Batch size for processing large Excel files
BATCH_SIZE: int = 100