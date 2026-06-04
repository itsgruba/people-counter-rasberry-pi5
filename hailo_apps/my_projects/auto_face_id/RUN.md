# Run Guide

This guide is for the combined `person -> face` prototype.

## What it does

- tracks only `person`
- finds a face inside each tracked person ROI
- removes non-`person` detections from the final ROI before display
- recognizes known faces
- enrolls unknown faces and assigns a `global_id`
- keeps `global_id` on the person box, not on the face box

## Video Stream

Use this stream as the input source:

```text
http://172.20.10.13:81/stream
```

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

## Notes

- If your person detector uses a different class ID, pass `--person-class-id <id>`.
- The main Hailo display should now show both `person` and `face` boxes.
- `--use-frame` is optional and only opens an extra debug window from Python.
- The app stores its database and samples in:

```text
my_projects/auto_face_id/database/
my_projects/auto_face_id/samples/
```

- The default resources for person detection, face detection, and face recognition are resolved automatically by the app.
