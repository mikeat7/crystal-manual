"""
Supabase Configuration for The Local Network

Stores Supabase connection details.
"""

import os

# Supabase credentials
SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "https://vcjxwlruvtstyvjytnaj.supabase.co"
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZjanh3bHJ1dnRzdHl2anl0bmFqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAyNzk5MzcsImV4cCI6MjA2NTg1NTkzN30.v6oFtLNM1me_LIY_XKyl58Ngt0Fm5qGiJAjBvhDzbgo"
)

# Default user ID for testing (can be overridden)
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default_user")

# Memory settings
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
ENABLE_CONVERSATION_MEMORY = os.getenv("ENABLE_CONVERSATION_MEMORY", "true").lower() == "true"
