# Auto Face ID

Automatic face enrollment and recognition for Hailo Apps.

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
- LanceDB for persistent `global_id` lookup

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
python3 my_projects/auto_face_id/person_face_id.py \
  --input http://172.20.10.13:81/stream \
  --width 320 \
  --height 240 \
  --disable-sync \
  --show-fps
```

## Storage

The app stores its own database and samples under:

```text
my_projects/auto_face_id/database/
my_projects/auto_face_id/samples/
```

It does not use or modify the standard `hailo_apps/python/pipeline_apps/face_recognition`
database.

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
