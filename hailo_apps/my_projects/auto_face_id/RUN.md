# Run Guide

This guide is for the combined `person -> face` prototype.

## What it does

- tracks only `person`
- finds a face inside each tracked person ROI
- removes non-`person` detections from the final ROI before display
- recognizes known faces
- enrolls unknown faces and assigns a `global_id`
- keeps the simple identity label such as `person_1` on the person box, not on the face box
- shows `Unknown` on a person box until an identity is assigned

## Video Stream

Use this stream as the input source:

```text
http://172.20.10.13:81/stream
```

## Run

From the repository root:

```bash
source setup_env.sh
python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --input http://172.20.10.13:81/stream \
  --width 320 \
  --height 240 \
  --disable-sync \
  --show-fps \
  --notify-url http://127.0.0.1:8000/api/events \
  --enroll-zone-file hailo_apps/my_projects/auto_face_id/enroll_zone.txt

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --input http://192.168.8.14:8080/stream \
  --width 640 \
  --height 640 \
  --disable-sync \
  --show-fps \
  --disable-local-display \
  --enroll-zone-file hailo_apps/my_projects/auto_face_id/enroll_zone.txt \
  --samples-per-person 3 \
  --unknown-sample-interval 2 \
  --min-unknown-age-seconds 0.5

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --input http://192.168.8.14:8080/stream \
  --width 640 \
  --height 640 \
  --disable-sync \
  --show-fps \
  --disable-local-display \
  --enroll-zone-file enroll_zone.txt \
  --notify-url http://192.168.8.6:8000/api/events \
  --samples-per-person 3 \
  --unknown-sample-interval 2 \
  --min-unknown-age-seconds 0.5

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --input http://192.168.8.14:8080/stream \
  --width 640 \
  --height 640 \
  --disable-sync \
  --show-fps \
  --disable-local-display \
  --enroll-zone-file hailo_apps/my_projects/auto_face_id/enroll_zone.txt \
  --notify-url http://192.168.8.6:8000/api/events \
  --samples-per-person 3 \
  --unknown-sample-interval 2 \
  --min-unknown-age-seconds 0.5

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --input http://192.168.8.6:8080/stream \
  --width 640 \
  --height 640 \
  --disable-sync \
  --show-fps \
  --disable-local-display \
  --enroll-zone-file hailo_apps/my_projects/auto_face_id/enroll_zone.txt \
  --notify-url http://192.168.8.6:8000/api/events \
  --samples-per-person 3 \
  --unknown-sample-interval 2 \
  --min-unknown-age-seconds 0.5

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --camera-mode entry \
  --input http://192.168.8.14:8080/stream \
  --width 640 \
  --height 640 \
  --disable-sync \
  --show-fps \
  --disable-local-display \
  --enroll-zone-file hailo_apps/my_projects/auto_face_id/enroll_zone.txt \
  --notify-url http://192.168.8.6:8000/api/events \
  --samples-per-person 3 \
  --unknown-sample-interval 2 \
  --min-unknown-age-seconds 0.5

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --camera-mode exit \
  --exit-recognition-zone-file exit_recognition_zone.txt \
  --input http://192.168.8.6:8080/stream \
  --width 640 \
  --height 640 \
  --disable-sync \
  --show-fps \
  --disable-local-display \
  --debug-stream-port 8091 \
  --notify-url http://192.168.8.6:8000/api/events

fastapi dev hailo_apps/my_projects/auto_face_id/person_face_api.py --host 0.0.0.0


python3 hailo_apps/my_projects/auto_face_id/stream.py

rasberry pi

ssh aleksandr@192.168.8.14 
python3 stream.py
```

## Notes

- If your person detector uses a different class ID, pass `--person-class-id <id>`.
- The main Hailo display should now show both `person` and `face` boxes.
- The face box should not show a person ID. Only the person box shows `Unknown` or `person_N`.
- `--use-frame` is optional and only opens an extra debug window from Python.
- New unknown identities are enrolled only from reasonably sharp, front-facing
  face samples. Tune this with `--min-enroll-blur-score`,
  `--max-enroll-nose-offset`, and `--min-enroll-eye-balance`.
- To create new people only inside a marked corridor, pass a normalized polygon
  with `--enroll-zone`, for example
  `--enroll-zone 0.35,0.35,0.65,0.35,0.90,1.0,0.10,1.0`. The debug stream draws
  the zone so you can adjust the points. Green foot markers are inside the zone;
  red foot markers are outside it.
- For manual tuning without restarting the app, pass
  `--enroll-zone-file hailo_apps/my_projects/auto_face_id/enroll_zone.txt`.
  Edit and save that file; the debug stream will show the updated polygon,
  vertex numbers, and entry lines. The same file can contain:
  `entry_line_a_y=0.55`, `entry_line_b_y=0.75`, and `entry_line_margin=0.02`.
- The entry lines are visible in the MJPEG debug stream at
  `http://<device-ip>:8090/debug`. A person is counted as `entered` after their
  foot point crosses line A and then line B.
- If people walk through the zone at normal speed, use faster enrollment:
  `--samples-per-person 3 --unknown-sample-interval 2 --min-unknown-age-seconds 0.5`.
- The app stores its database and samples in:

```text
hailo_apps/my_projects/auto_face_id/database/persons.sqlite3
hailo_apps/my_projects/auto_face_id/samples/
```

- Start the FastAPI backend with:

```bash
uvicorn hailo_apps.my_projects.auto_face_id.person_face_api:app --host 127.0.0.1 --port 8000
```

- The two current lists are available at:

```text
http://127.0.0.1:8000/api/people
http://127.0.0.1:8000/api/entered-people
```

- The default resources for person detection, face detection, and face recognition are resolved automatically by the app.
- Inspect the current people with `python3 hailo_apps/my_projects/auto_face_id/inspect_database.py`.
