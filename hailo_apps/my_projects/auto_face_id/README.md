# Auto Face ID

Automatic face enrollment and recognition for Hailo Apps.

The project uses a local SQLite database for identities and NumPy cosine similarity for
face embedding search.

## Plan

Current working plan: [PLAN.md](./PLAN.md)

## New Prototype

The new combined pipeline entrypoint is [person_face_id.py](./person_face_id.py).
It runs face detection/recognition and person detection/tracking on Hailo,
then uses the face only as a signal to recognize the person track.
The visual identity is attached to the `person` box, which is the source of truth.

## Run Guide

Use [RUN.md](./RUN.md) for the exact launch command with the live stream input.

This app uses the existing Hailo face-recognition pipeline:

- SCRFD for face detection
- ArcFace MobileFaceNet for face embeddings
- SQLite for persistent people, embeddings, and sample paths

New faces are enrolled automatically from live video. The tracker `track_id` is temporary, but
the printed `global_id` is persistent across disappear/reappear events when recognition matches.
If a face cannot be matched confidently to a person ROI, the app waits instead of assigning an ID
too early.

The display stays intentionally simple:

- the face box is labeled only as `face`
- an unidentified person box is labeled `Unknown`
- a recognized or newly enrolled person box is labeled with a simple ID such as `person_1`

## Run

From the repository root:

```bash
source setup_env.sh
python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --input http://172.20.10.13:81/stream \
  --width 320 \
  --height 240 \
  --disable-sync \
  --show-fps

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --input http://192.168.8.10:81/stream \
  --width 320 \
  --height 240 \
  --disable-sync \
  --show-fps

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --input http://192.168.8.14:8080/stream \
  --width 320 \
  --height 240 \
  --disable-sync \
  --show-fps
```

## Storage

The app stores its own database and samples under:

```text
hailo_apps/my_projects/auto_face_id/database/persons.sqlite3
hailo_apps/my_projects/auto_face_id/samples/
```

It does not use or modify the standard `hailo_apps/python/pipeline_apps/face_recognition`
database. The old project-local LanceDB folder `database/persons.db/` is no longer used after
migration.

To migrate an existing project-local LanceDB again, run:

```bash
venv_hailo_apps/bin/python hailo_apps/my_projects/auto_face_id/migrate_lancedb_to_sqlite.py
```

## Inspect SQLite

Open `database/persons.sqlite3` in VS Code with a SQLite viewer extension, or run:

```bash
python3 hailo_apps/my_projects/auto_face_id/inspect_database.py
```

The database contains:

- `persons`: one row per known person
- `face_samples`: one row per face embedding and JPEG sample

## Maintenance CLI

For day-to-day cleanup, use the dedicated maintenance script. It always removes the database
rows and the matching sample files together, so the SQLite data and `samples/` folder stay in
sync.

```bash
python3 hailo_apps/my_projects/auto_face_id/manage_database.py inspect
python3 hailo_apps/my_projects/auto_face_id/manage_database.py delete-person --label person_1
python3 hailo_apps/my_projects/auto_face_id/manage_database.py delete-person --global-id <uuid>
python3 hailo_apps/my_projects/auto_face_id/manage_database.py clear-all
python3 hailo_apps/my_projects/auto_face_id/manage_database.py prune-samples
python3 hailo_apps/my_projects/auto_face_id/manage_database.py repair
```

If the package is installed, the same CLI is available as:

```bash
hailo-auto-face-db inspect
```

## Dashboard API

For a small frontend, run the FastAPI backend:

```bash
uvicorn hailo_apps.my_projects.auto_face_id.person_face_api:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

It exposes:

- `GET /api/people` - lightweight карточки людей with `label`, `visit_count`, `thumbnail_url`
- `GET /api/people/{global_id}` - full record for one person
- `DELETE /api/people/{global_id}` - delete one person and their saved sample files
- `GET /samples/{filename}` - serves the saved JPEG sample images
- `POST /api/events` - optional event sink for camera notifications

Manual deletion example:

```bash
curl -X DELETE http://127.0.0.1:8000/api/people/<uuid>
```

The frontend can poll `GET /api/people` every few seconds, or you can use
`--notify-url http://127.0.0.1:8000/api/events` in `person_face_id.py` to push events on
recognition/enrollment.

The visit counter follows the `track_id` session rule:

- if the same person stays on the same `track_id`, `visit_count` does not change
- if the person leaves and comes back with a new `track_id`, `visit_count` increases
- the current `last_seen_track_id` is stored in SQLite so the backend and frontend can stay in sync

Useful SQL queries:

```sql
SELECT COUNT(*) FROM persons;

SELECT id, label, global_id FROM persons ORDER BY id;

SELECT persons.label, COUNT(face_samples.id) AS samples
FROM persons
LEFT JOIN face_samples ON face_samples.person_id = persons.id
GROUP BY persons.id
ORDER BY persons.id;

SELECT label, visits_count AS visit_count, last_seen_at, last_seen_track_id
FROM persons
ORDER BY id;
```

Default resources are resolved automatically for:

- person detection
- face detection
- face recognition

## Behavior

When a face is recognized:

```text
recognized: track_id=4 global_id=<uuid> label=person_1 confidence=0.82
```

When a new face is not recognized, the app collects three face samples:

```text
collecting: track_id=5 samples=1/3
collecting: track_id=5 samples=2/3
collecting: track_id=5 samples=3/3
enrolled: track_id=5 global_id=<uuid> label=person_2 confidence=1.00
```

Tune enrollment with:

```bash
--samples-per-person 3
--unknown-sample-interval 5
--min-enroll-confidence 0.55
```
