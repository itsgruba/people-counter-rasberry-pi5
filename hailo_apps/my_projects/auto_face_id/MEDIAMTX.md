# MediaMTX: камера → распознавание → debug

В этом checkout приложение — файл `person_face_id.py`, а не каталог.

## Запуск на Raspberry Pi

1. Остановите старый `stream.py`: камерой должен владеть один процесс.
2. Если `/cam` уже работает через MediaMTX, сохраните его настройки. Добавьте
   в существующий раздел `paths` пути `debug` и `debug-exit` с `source: publisher`.
   Для нового запуска приложен `mediamtx.example.yml` (CSI-камера, квадратный
   кадр как у прежнего stream.py, 15 FPS). Запустите установленный MediaMTX:
   `mediamtx hailo_apps/my_projects/auto_face_id/mediamtx.example.yml`.
   Не запускайте второй сервер, если MediaMTX уже работает как служба.
3. Установите FFmpeg: `sudo apt install ffmpeg`. Проверьте наличие кодировщика:
   `ffmpeg -hide_banner -h encoder=libx264`.
4. Из корня проекта, после `source setup_env.sh`:

```bash
source /home/aleksandr/hailo-apps/venv_hailo_apps/bin/activate

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --camera-mode exit \
  --input rtsp://127.0.0.1:8554/cam \
  --width 1920 --height 1080 --frame-rate 15 \
  --disable-sync --disable-local-display \
  --debug-stream-transport rtsp \
  --debug-rtsp-url rtsp://127.0.0.1:8554/debug_rpi5 \
  --debug-stream-fps 8 --debug-stream-width 960 --debug-bitrate 1200 \
  --notify-url http://127.0.0.1:8000/api/events \
  --exit-recognition-zone-file hailo_apps/my_projects/auto_face_id/exit_recognition_zone.txt

python3 hailo_apps/my_projects/auto_face_id/person_face_id.py \
  --camera-mode entry \
  --input rtsp://192.168.0.3:8554/cam \
  --width 1920 --height 1080 --frame-rate 15 \
  --disable-sync --disable-local-display \
  --debug-stream-transport rtsp \
  --debug-rtsp-url rtsp://127.0.0.1:8554/debug_zero \
  --debug-stream-fps 8 --debug-stream-width 960 --debug-bitrate 1200 \
  --notify-url http://127.0.0.1:8000/api/events \
  --enroll-zone-file hailo_apps/my_projects/auto_face_id/enroll_zone.txt
```

Добавьте прежние параметры уведомлений, регистрации и счётчиков из вашей
команды запуска. Логика распознавания, SQLite, зон и уведомлений не менялась.
`start_all.sh` сохраняет прежние параметры ENTRY/EXIT, но использует RTSP и больше
не запускает stream.py. Он предполагает уже запущенный MediaMTX. Адреса входов
можно переопределить переменными ENTRY_INPUT и EXIT_INPUT. EXIT публикуется в
`/debug-exit`, чтобы два приложения не конкурировали за `/debug`.

Просмотр в VLC: `rtsp://IP_ВАШЕЙ_PI:8554/debug`. На самой Pi можно использовать
127.0.0.1; на другом компьютере этот адрес указывает на сам компьютер.
RTSP напрямую в HTML `<img>` не работает. Для старого интерфейса и `/health`
выберите `--debug-stream-transport http --debug-stream-port 8090`.
`--disable-debug-stream` отключает любой debug-вывод. `--use-frame` для RTSP
не нужен: он дополнительно включает прежнюю передачу кадров локальному UI.

## Что изменилось и чего ожидать

- Старый stream.py делает JPEG 1640×1640 с качеством 90 для каждого HTTP-клиента.
  В новой схеме он не участвует: исходным потоком управляет MediaMTX.
- JPEG debug ранее кодировался синхронно в callback. RTSP использует отдельный
  поток и FFmpeg/libx264 ultrafast/zerolatency, максимум два потока кодировщика.
- Debug рисуется и отправляется только с заданной частотой, по умолчанию 10 FPS.
  `--debug-stream-width 960` ограничивает ширину debug, сохраняя пропорции
  (1920×1080 → 960×540). По умолчанию 0: исходный размер, увеличения нет.
  Размер кадра распознавания и сохраняемых снимков не меняется. В headless-режиме
  уменьшение выполняется до отрисовки; H.264 кодируется уже в меньшем размере.
  Для начала используйте 8 FPS и 1200 кбит/с; фактический выигрыш измерьте на Pi.
- Есть только один ожидающий кадр. При зависшем выводе запись прерывается через
  секунду, FFmpeg перезапускается с паузой 2 секунды. Распознавание не ждёт сеть.
- RTSP читает по TCP с jitter-buffer 100 мс (`--rtsp-latency-ms`). Очередь перед
  декодером не сбрасывает сжатые пакеты; после декодирования действует прежняя
  политика коротких очередей. На нестабильной сети попробуйте 200–300 мс.
- H.264 на Pi5 кодируется программно. MediaMTX сам по себе не делает обработку
  бесплатной; две камеры означают два декодирования и два Hailo pipeline.
  Для дополнительной экономии уменьшайте FPS камеры/обработки и debug FPS.
- Смена разрешения/соотношения сторон или кадрирования камеры может потребовать
  повторной настройки зон. Не заменяйте уже удачную конфигурацию `/cam` вслепую.

## Проверка на оборудовании

Сравните одинаковые входное разрешение, сцену и FPS: старый HTTP, RTSP без debug,
RTSP с debug. Используйте `top -H`, `pidstat -u 1` (пакет sysstat), `--show-fps`;
следите за температурой и throttling (`vcgencmd measure_temp`, `vcgencmd get_throttled`).
Проверьте задержку по часам в кадре, качество лиц, регистрацию, пересечение зон,
уведомления и просмотр несколькими клиентами. Перезапустите MediaMTX и убедитесь,
что debug восстанавливается. Восстановление входного RTSP остаётся ответственностью
существующего GStreamerApp и требует отдельной проверки на вашей сборке.

Локальная проверка на macOS не заменяет запуск с Hailo, GStreamer и камерой на Pi.

Источники: [камеры Raspberry Pi в MediaMTX](https://mediamtx.org/docs/publish/raspberry-pi-cameras),
[публикация RTSP](https://mediamtx.org/docs/publish/rtsp-clients),
[H.264 на Pi5](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010033-WP-1-H.264%20encoding%20performance%20on%20Raspberry%20Pi%205_series%20computers.pdf).


Перед запуском камер запустите backend на той же Pi и из той же копии проекта:

```bash
python3 hailo_apps/my_projects/auto_face_id/person_face_api.py --host 0.0.0.0 --port 8000
```

`--notify-url http://127.0.0.1:8000/api/events` в обеих командах включает
уведомления backend для WebSocket. Без этого параметра HTTP читает БД, но
обновления от камер по WebSocket не отправляются. `start_all.sh` запускает API
без development reload и использует этот локальный адрес (переопределяется `NOTIFY_URL`).
