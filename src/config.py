from __future__ import annotations
import os

# Default to local API; you can override via ENV or st.secrets later
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
