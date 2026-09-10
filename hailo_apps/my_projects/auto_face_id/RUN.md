> Новый рекомендуемый запуск через MediaMTX: [MEDIAMTX.md](MEDIAMTX.md).
> Debug по умолчанию теперь RTSP `/debug`; для HTTP-команд ниже добавьте
> `--debug-stream-transport http`.

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
http://127.0.0.1:8000/api/exits?limit=50&offset=0
```

- The default resources for person detection, face detection, and face recognition are resolved automatically by the app.
- Inspect the current people with `python3 hailo_apps/my_projects/auto_face_id/inspect_database.py`.

### Линии A/B по двум точкам и контур зоны

В `enroll_zone.txt` задайте отрезки входа:

```ini
entry_line_a=0.10,0.55,0.80,0.70
entry_line_b=0.10,0.75,0.80,0.90
entry_line_margin=0.02
```

В `exit_recognition_zone.txt` полигон распознавания и отрезки выхода задаются отдельно:

```ini
0.01,0.5,0.3,0.5,0.75,0.95,0.08,0.95
exit_line_a=0.10,0.55,0.80,0.70
exit_line_b=0.10,0.75,0.80,0.90
exit_line_margin=0.02
```

Формат линии — `x1,y1,x2,y2`, координаты от 0 до 1; начало координат
в левом верхнем углу. Поддерживаются горизонтальные, наклонные и вертикальные
отрезки. Считается пересечение именно отрезка нижней центральной точкой рамки
человека, в порядке A → B. Для смены порядка поменяйте местами A и B.
`margin` — перпендикулярное расстояние до линии в нормализованных координатах;
внутри этой полосы смена стороны не фиксируется. Старые ключи `*_line_a_y`
и `*_line_b_y` продолжают задавать горизонтальные линии на всю ширину кадра.
Не задавайте одну линию одновременно новым и старым ключом.

Файлы перечитываются при работе с `--enroll-zone-file` и
`--exit-recognition-zone-file`. При изменении геометрии история пересечений
сбрасывается; неверные параметры сохраняют предыдущую рабочую конфигурацию.
В поставляемых файлах сохранено прежнее положение линий, записанное новым форматом.

В режиме `--camera-mode exit` debug-видео отображает замкнутый фиолетовый контур
`exit recognition zone` из строки полигона, его вершины и линии EXIT A/B.
Контур рисуется поверх рамок детекций. Для него нужен
`--exit-recognition-zone-file exit_recognition_zone.txt`.
Он отображается в обработанном debug-потоке; исходный поток камеры не содержит
этой разметки. Для локального окна включите `--use-frame`.

### Новая запись на каждый вход

Камера ENTRY больше не сравнивает лицо с историей людей. Она собирает образцы
лица для текущего трека и создаёт новую запись с новым `global_id` после A → B.
Повторный приход того же человека с новым треком создаёт отдельную запись.
До пересечения запись не создаётся; при недостатке образцов остаётся механизм
отложенного входа с созданием placeholder по таймауту. Если счётчик входа явно
отключён, регистрация происходит после накопления образцов без ожидания A → B.

EXIT по-прежнему сопоставляет лицо только с теми, кто сейчас находится внутри.
После выхода запись и фотографии остаются в истории, но не используются для
распознавания следующего входа. Признаки лица на ENTRY всё ещё вычисляются:
они нужны для сопоставления этого входа с выходом. Команды запуска менять не нужно.

Фото отложенного входа теперь копируется в память при пересечении B: сохраняется
область человека с существующим отступом, а при пустой области — весь кадр.
При создании записи используется именно этот снимок, даже если человек уже исчез
из детекций. После обработки контекст удаляется и копия освобождается. Сбор
качественных образцов лица и таймаут регистрации не меняются. Копия находится
в оперативной памяти и не переживает остановку процесса; повторов при ошибках
диска это изменение не добавляет.


### Backend: список выходов и обновления

`GET /api/exits?limit=50&offset=0` заменяет удалённый `GET /api/visit-events`.
Одна запись — один выход с соответствующим входом. Новые выходы сверху.
`limit`: 1–1000 (по умолчанию 50), `offset`: от 0. `total` — общее число выходов,
а не размер страницы. Для следующей страницы увеличьте `offset` на `limit`.

```json
{
  "items": [{
    "id": "exit-event-id",
    "global_id": "admission-person-id",
    "label": "person_1",
    "visit_number": 1,
    "entered_at": 1789020000,
    "exited_at": 1789023600,
    "duration_seconds": 3600,
    "entry_photo_url": "http://PI_IP:8000/samples/person_1/entry.jpeg",
    "exit_photo_url": "http://PI_IP:8000/samples/person_1/exit.jpeg"
  }],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

Если соответствующий вход отсутствует в старых данных, `entered_at`,
`duration_seconds` и `entry_photo_url` будут `null`; выход остаётся в списке.
Связь определяется по человеку и хронологии: берётся последний вход после
предыдущего выхода. Фото от уже завершённого посещения не переиспользуется.
Существующие события в БД сохраняются. Подробная карточка человека по-прежнему
содержит `visit_events`; отдельного HTTP-маршрута журнала больше нет.

`GET /api/entered-people?limit=100` возвращает текущих людей внутри и `total_inside`,
не загружая историю других людей. `GET /api/people` получает карточки без загрузки
векторов лиц и всей истории. Если образца лица нет, миниатюрой карточки становится
последняя фотография входа.

В сообщениях WebSocket `init`, `inside` и `outside` поле `data.visit_events`
заменено на `data.exits`: тот же объект пагинации, что HTTP, первые 50 записей.
При событии можно обновить текущую страницу отдельным GET-запросом. Уведомления
работают при запущенном API и `--notify-url http://127.0.0.1:8000/api/events`
в командах ОБЕИХ камер. API и камеры должны работать с общей SQLite-базой и папкой
снимков; для текущей схемы запускайте их из одной копии проекта на RPi5.
Без уведомлений HTTP работает, но автоматические обновления камер не приходят.
После недоступности API перечитайте состояние; доставка уведомлений best-effort,
без очереди повторов. Команды камер приведены в `MEDIAMTX.md`.
