#!/bin/bash

cd "$(dirname "$0")" || exit 1

source setup_env.sh

mkdir -p logs

cleanup() {
    echo "Stopping processes..."
    kill "$API_PID" 2>/dev/null
    kill "$ENTRY_PID" 2>/dev/null
    kill "$EXIT_PID" 2>/dev/null
}

trap cleanup EXIT INT TERM

# MediaMTX must already be running on each camera host; it owns the camera.

echo "Starting API..."

python3 \
    hailo_apps/my_projects/auto_face_id/person_face_api.py \
    --host 0.0.0.0 \
    >> logs/api.log 2>&1 &

API_PID=$!

sleep 3


echo "Starting ENTRY..."

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
    --camera-mode entry \
    --input "${ENTRY_INPUT:-rtsp://127.0.0.1:8554/cam}" \
    --width 640 \
    --height 640 \
    --disable-sync \
    --show-fps \
    --disable-local-display \
    --enroll-zone-file hailo_apps/my_projects/auto_face_id/enroll_zone.txt \
    --notify-url "${NOTIFY_URL:-http://127.0.0.1:8000/api/events}" \
    --samples-per-person 3 \
    --unknown-sample-interval 2 \
    --min-unknown-age-seconds 0.5 \
    >> logs/entry.log 2>&1 &

ENTRY_PID=$!


echo "Starting EXIT..."

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
    --camera-mode exit \
    --exit-recognition-zone-file hailo_apps/my_projects/auto_face_id/exit_recognition_zone.txt \
    --input "${EXIT_INPUT:-rtsp://192.168.0.3:8554/cam}" \
    --width 640 \
    --height 640 \
    --disable-sync \
    --show-fps \
    --disable-local-display \
    --debug-rtsp-url rtsp://127.0.0.1:8554/debug-exit \
    --notify-url "${NOTIFY_URL:-http://127.0.0.1:8000/api/events}" \
    >> logs/exit.log 2>&1 &

EXIT_PID=$!

echo "All processes started."

wait
