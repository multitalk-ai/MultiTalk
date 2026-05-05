# List of vLLM OpenAI-compatible endpoints used by the load-balanced Gemma backend
# (--gemma4 --gemma4_list). Replace the placeholders with your own served endpoints.
# The base_url should be the vLLM "/v1" path, e.g. "http://10.0.0.1:8000/v1".
api_bases = [
    "http://127.0.0.1:8000/v1",
    # "http://127.0.0.1:8001/v1",
    # "http://127.0.0.1:8002/v1",
    # ...
]
