#!/usr/bin/env python3
"""FastAPI backend for whole-person body ReID.

This module deliberately reuses the complete person_face_api application so
HTTP routes, response schemas, event handling, and WebSocket messages stay
identical. Only the default SQLite database and sample directory differ.
"""

from __future__ import annotations

import logging
import os

# person_face_api reads these settings while it is imported. Set body-safe
# defaults first so face and body embeddings can never be mixed accidentally.
os.environ.setdefault("PERSON_ID_DB_NAME", "persons_body.sqlite3")
os.environ.setdefault("PERSON_ID_SAMPLES_DIR", "body_samples")

from hailo_apps.my_projects.auto_face_id.person_face_api import (  # noqa: E402
    _uvicorn_import_error,
    app,
    build_parser,
    uvicorn,
)


def main() -> None:
    args = build_parser().parse_args()
    if uvicorn is None:  # pragma: no cover - only hit when runtime deps are incomplete
        raise RuntimeError(
            "uvicorn is required to run this server. Install it with `pip install uvicorn`."
        ) from _uvicorn_import_error

    uvicorn.run(
        "hailo_apps.my_projects.auto_face_id.person_body_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
