"""
TuneForge database package.

Re-exports the core Database class, ORM models, CRUD operations,
and helper functions for use throughout the API.
"""

from tuneforge.api.database.engine import Database, total_pages
from tuneforge.api.database.models import (
    ApiKeyRow,
    Base,
    CreditRow,
    CreditTransactionRow,
    GenerationRow,
    TrackRow,
    UserRow,
    row_to_scores,
)
from tuneforge.api.database.crud import (
    create_api_key,
    create_credit,
    create_credit_transaction,
    create_generation,
    create_track,
    create_user,
    delete_track,
    get_api_key_by_hash,
    get_credit,
    get_credit_transactions,
    get_generation,
    get_track,
    get_track_count,
    get_tracks_by_generation,
    get_user_api_keys,
    get_user_by_email,
    get_user_by_google_id,
    get_user_by_id,
    revoke_api_key,
    search_tracks,
    update_api_key_last_used,
    update_generation_status,
    update_user,
)

__all__ = [
    # Engine
    "Database",
    "total_pages",
    # Base
    "Base",
    # Models
    "ApiKeyRow",
    "CreditRow",
    "CreditTransactionRow",
    "GenerationRow",
    "TrackRow",
    "UserRow",
    # Helpers
    "row_to_scores",
    # CRUD - Tracks
    "create_track",
    "delete_track",
    "get_track",
    "search_tracks",
    "get_track_count",
    "get_tracks_by_generation",
    # CRUD - Users
    "create_user",
    "get_user_by_email",
    "get_user_by_id",
    "get_user_by_google_id",
    "update_user",
    # CRUD - Credits
    "create_credit",
    "get_credit",
    "create_credit_transaction",
    "get_credit_transactions",
    # CRUD - Generations
    "create_generation",
    "get_generation",
    "update_generation_status",
    # CRUD - API Keys
    "create_api_key",
    "get_api_key_by_hash",
    "get_user_api_keys",
    "revoke_api_key",
    "update_api_key_last_used",
]
