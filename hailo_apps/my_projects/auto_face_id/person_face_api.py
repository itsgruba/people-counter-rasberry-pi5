#!/usr/bin/env python3
"""FastAPI backend for the person-face dashboard."""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import uvicorn
except ImportError as exc:  # pragma: no cover - only hit when runtime deps are incomplete
    uvicorn = None  # type: ignore[assignment]
    _uvicorn_import_error = exc
else:
    _uvicorn_import_error = None

try:
    from hailo_apps.my_projects.auto_face_id.sqlite_db_handler import SQLiteDatabaseHandler
except ImportError:
    from sqlite_db_handler import SQLiteDatabaseHandler


PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_DIR / "database"
SAMPLES_DIR = PROJECT_DIR / "samples"
DB_NAME = "persons.sqlite3"

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    ok: bool = Field(..., examples=[True])


class PersonCard(BaseModel):
    global_id: str = Field(..., examples=["3f0d7e6f-2a8f-4f8a-98f7-0f4f7f73c2f1"])
    label: str = Field(..., examples=["person_1"])
    visit_count: int = Field(..., examples=[4])
    last_seen_at: int | None = Field(default=None, examples=[1717500000])
    thumbnail_name: str | None = Field(default=None, examples=["abc123.jpeg"])
    thumbnail_url: str | None = Field(
        default=None,
        examples=["http://127.0.0.1:8000/samples/abc123.jpeg"],
    )
    sample_count: int = Field(..., examples=[5])


class PeopleResponse(BaseModel):
    people: list[PersonCard]
    count: int = Field(..., examples=[1])


class SampleRef(BaseModel):
    id: str | None = Field(default=None, examples=["sample-1"])
    timestamp: int | None = Field(default=None, examples=[1717500000])
    sample_url: str | None = Field(
        default=None,
        examples=["http://127.0.0.1:8000/samples/abc123.jpeg"],
    )


class PersonDetails(BaseModel):
    global_id: str = Field(..., examples=["3f0d7e6f-2a8f-4f8a-98f7-0f4f7f73c2f1"])
    label: str = Field(..., examples=["person_1"])
    visit_count: int = Field(..., examples=[4])
    last_seen_at: int | None = Field(default=None, examples=[1717500000])
    last_seen_track_id: int | None = Field(default=None, examples=[12])
    samples: list[SampleRef]


class PersonResponse(BaseModel):
    person: PersonDetails


class DeletePersonResponse(BaseModel):
    ok: bool = Field(..., examples=[True])
    global_id: str = Field(..., examples=["3f0d7e6f-2a8f-4f8a-98f7-0f4f7f73c2f1"])
    label: str = Field(..., examples=["person_1"])
    deleted_sample_count: int = Field(..., examples=[5])


class EventPayload(BaseModel):
    event: str = Field(..., examples=["recognized"])
    global_id: str = Field(..., examples=["3f0d7e6f-2a8f-4f8a-98f7-0f4f7f73c2f1"])
    label: str = Field(..., examples=["person_1"])
    track_id: int = Field(..., examples=[12])
    confidence: float = Field(..., examples=[0.91])
    visit_count: int | None = Field(default=None, examples=[4])
    timestamp: int = Field(..., examples=[1717500000])


class EventAck(BaseModel):
    ok: bool = Field(..., examples=[True])


class SocketMessage(BaseModel):
    type: str = Field(..., examples=["init"])
    data: dict[str, Any]


HEALTH_EXAMPLE = {"ok": True}
PEOPLE_EXAMPLE = {
    "people": [
        {
            "global_id": "3f0d7e6f-2a8f-4f8a-98f7-0f4f7f73c2f1",
            "label": "person_1",
            "visit_count": 4,
            "last_seen_at": 1717500000,
            "thumbnail_name": "abc123.jpeg",
            "thumbnail_url": "http://127.0.0.1:8000/samples/abc123.jpeg",
            "sample_count": 5,
        }
    ],
    "count": 1,
}
PERSON_EXAMPLE = {
    "person": {
        "global_id": "3f0d7e6f-2a8f-4f8a-98f7-0f4f7f73c2f1",
        "label": "person_1",
        "visit_count": 4,
        "last_seen_at": 1717500000,
        "last_seen_track_id": 12,
        "samples": [
            {
                "id": "sample-1",
                "timestamp": 1717500000,
                "sample_url": "http://127.0.0.1:8000/samples/abc123.jpeg",
            }
        ],
    }
}
EVENT_EXAMPLE = {
    "event": "recognized",
    "global_id": "3f0d7e6f-2a8f-4f8a-98f7-0f4f7f73c2f1",
    "label": "person_1",
    "track_id": 12,
    "confidence": 0.91,
    "visit_count": 4,
    "timestamp": 1717500000,
}
DELETE_PERSON_EXAMPLE = {
    "ok": True,
    "global_id": "3f0d7e6f-2a8f-4f8a-98f7-0f4f7f73c2f1",
    "label": "person_1",
    "deleted_sample_count": 5,
}

WEBSOCKET_EVENT_EXAMPLE = {
    "type": "recognized",
    "data": EVENT_EXAMPLE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve face-ID data for a frontend dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload mode for development.",
    )
    return parser


def _create_db() -> SQLiteDatabaseHandler:
    return SQLiteDatabaseHandler(
        db_name=DB_NAME,
        threshold=0.55,
        database_dir=DATABASE_DIR,
        samples_dir=SAMPLES_DIR,
    )


def _sample_url(sample_path: str | None) -> str | None:
    if not sample_path:
        return None
    return f"/samples/{Path(sample_path).name}"


def _absolute_url(request: Request, path: str | None) -> str | None:
    if not path:
        return None
    return str(request.base_url).rstrip("/") + path


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._connections)

        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:  # noqa: BLE001
                await self.disconnect(websocket)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = _create_db()
    app.state.ws_manager = WebSocketManager()
    try:
        yield
    finally:
        app.state.db.close()


app = FastAPI(
    title="Person Face API",
    version="1.0.0",
    description="Backend for listing people, sample images, and visit counters.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


def _db(request: Request) -> SQLiteDatabaseHandler:
    return request.app.state.db


def _ws_manager(scope: Any) -> WebSocketManager:
    return scope.app.state.ws_manager


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={200: {"content": {"application/json": {"example": HEALTH_EXAMPLE}}}},
)
def health() -> HealthResponse:
    return {"ok": True}


@app.get(
    "/api/people",
    response_model=PeopleResponse,
    responses={200: {"content": {"application/json": {"example": PEOPLE_EXAMPLE}}}},
)
def list_people(request: Request) -> PeopleResponse:
    db = _db(request)
    people = db.get_people_cards()
    for person in people:
        person["thumbnail_url"] = _absolute_url(
            request,
            _sample_url(person.get("thumbnail_name")),
        )
    return {"people": people, "count": len(people)}


@app.get(
    "/api/people/{global_id}",
    response_model=PersonResponse,
    responses={200: {"content": {"application/json": {"example": PERSON_EXAMPLE}}}},
)
def get_person(global_id: str, request: Request) -> PersonResponse:
    db = _db(request)
    person = db.get_record_by_id(global_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    samples = []
    for sample in person.get("samples_json") or []:
        samples.append(
            {
                "id": sample.get("id"),
                "timestamp": sample.get("timestamp"),
                "sample_url": _absolute_url(request, _sample_url(sample.get("sample_path"))),
            }
        )

    return {
        "person": {
            "global_id": person["global_id"],
            "label": person["label"],
            "visit_count": person["visit_count"],
            "last_seen_at": person.get("last_seen_at"),
            "last_seen_track_id": person.get("last_seen_track_id"),
            "samples": samples,
        }
    }


@app.delete(
    "/api/people/{global_id}",
    response_model=DeletePersonResponse,
    responses={
        200: {"content": {"application/json": {"example": DELETE_PERSON_EXAMPLE}}},
        404: {"description": "Person not found"},
    },
)
async def delete_person(global_id: str, request: Request) -> DeletePersonResponse:
    db = _db(request)
    person = db.get_record_by_id(global_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    deleted_sample_count = len(person.get("samples_json") or [])
    db.delete_record(global_id)

    response = {
        "ok": True,
        "global_id": person["global_id"],
        "label": person["label"],
        "deleted_sample_count": deleted_sample_count,
    }
    await _ws_manager(request).broadcast(
        {
            "type": "person_deleted",
            "data": response,
        }
    )
    return response


@app.get("/samples/{filename}")
def get_sample(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = (SAMPLES_DIR / safe_name).resolve()
    try:
        file_path.relative_to(SAMPLES_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Forbidden") from exc

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")

    return FileResponse(file_path)


@app.post(
    "/api/events",
    response_model=EventAck,
    responses={200: {"content": {"application/json": {"example": {"ok": True}}}}},
)
async def receive_event(payload: EventPayload, request: Request) -> EventAck:
    logger.info("Event received: %s", payload)
    await _ws_manager(request).broadcast(
        {
            "type": payload.event,
            "data": payload.model_dump(),
        }
    )
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await _ws_manager(websocket).connect(websocket)
    try:
        await websocket.send_json(
            {
                "type": "init",
                "data": {
                    "people": websocket.app.state.db.get_people_cards(),
                },
            }
        )
        while True:
            # Keep the connection alive and optionally consume client messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await _ws_manager(websocket).disconnect(websocket)


def main() -> None:
    args = build_parser().parse_args()
    if uvicorn is None:  # pragma: no cover - only hit when runtime deps are incomplete
        raise RuntimeError(
            "uvicorn is required to run this server. Install it with `pip install uvicorn`."
        ) from _uvicorn_import_error

    uvicorn.run(
        "hailo_apps.my_projects.auto_face_id.person_face_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
