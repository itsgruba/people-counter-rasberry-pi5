"""API integration tests against an isolated SQLite database and sample directory."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi.staticfiles import StaticFiles
from hailo_apps.my_projects.auto_face_id import person_face_api as api
from hailo_apps.my_projects.auto_face_id.sqlite_db_handler import SQLiteDatabaseHandler


class ApiExitsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.samples = self.root / 'samples'
        self.db = SQLiteDatabaseHandler('test.sqlite3', .55, self.root, self.samples)
        for name, value in [('SAMPLES_DIR', self.samples), ('_create_db', lambda: self.db)]:
            patcher = patch.object(api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        mount = next(r for r in api.app.routes if getattr(r, 'name', None) == 'samples')
        patcher = patch.object(mount, 'app', StaticFiles(directory=str(self.samples)))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = self.enterContext(TestClient(api.app))

    def person(self, name):
        return self.db.create_placeholder_record(timestamp=1, label=name)['global_id']

    def event(self, person, kind, timestamp, photo, visit=1):
        path = self.samples / photo if photo else None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'test-photo')
        return self.db.add_visit_event(person, kind, visit, timestamp, str(path) if path else '')

    def test_pairing_pagination_and_photos(self):
        person = self.person('A')
        self.event(person, 'entry', 10, 'A/in1.jpg')
        self.event(person, 'exit', 20, 'A/out1.jpg', visit=2)
        self.event(person, 'entry', 30, 'A/in2.jpg', visit=3)
        self.event(person, 'exit', 40, 'A/out2.jpg', visit=4)
        self.event(self.person('B'), 'entry', 50, 'B/in.jpg')
        result = self.client.get('/api/exits?limit=1').json()
        self.assertEqual(result['total'], 2)
        self.assertEqual(len(result['items']), 1)
        item = result['items'][0]
        self.assertEqual((item['entered_at'], item['exited_at'], item['duration_seconds']), (30, 40, 10))
        self.assertTrue(item['entry_photo_url'].endswith('/A/in2.jpg'))
        self.assertEqual(self.client.get(item['entry_photo_url']).status_code, 200)
        self.assertEqual(self.client.get(item['exit_photo_url']).status_code, 200)
        older = self.client.get('/api/exits?limit=1&offset=1').json()['items'][0]
        self.assertEqual(older['entered_at'], 10)
        empty = self.client.get('/api/exits?offset=100').json()
        self.assertEqual(empty['items'], [])
        self.assertEqual(empty['total'], 2)

    def test_missing_entry_never_borrows_previous_photo(self):
        person = self.person('A')
        self.event(person, 'entry', 10, 'in.jpg')
        self.event(person, 'exit', 20, 'out.jpg')
        self.event(person, 'exit', 30, '')
        item = self.client.get('/api/exits').json()['items'][0]
        for key in ('entered_at', 'duration_seconds', 'entry_photo_url', 'exit_photo_url'):
            self.assertIsNone(item[key])

    def test_equal_timestamps_are_ordered_by_event_insertion(self):
        person = self.person('A')
        self.event(person, 'entry', 10, 'in1.jpg')
        self.event(person, 'exit', 10, 'out1.jpg')
        self.event(person, 'entry', 10, 'in2.jpg')
        self.event(person, 'exit', 10, 'out2.jpg')
        items = self.client.get('/api/exits').json()['items']
        self.assertTrue(items[0]['entry_photo_url'].endswith('/in2.jpg'))
        self.assertTrue(items[1]['entry_photo_url'].endswith('/in1.jpg'))

    def test_inside_thumbnail_and_no_global_history_queries(self):
        person = self.person('A')
        self.event(person, 'entry', 10, 'A/in.jpg')
        self.db.set_person_inside(person, 10)
        with patch.object(self.db, 'get_visit_events', side_effect=AssertionError('whole history')), patch.object(self.db, 'get_people_cards', side_effect=AssertionError('all people')):
            result = self.client.get('/api/entered-people?limit=1').json()
        self.assertEqual(result['total_inside'], 1)
        self.assertTrue(result['entered_people'][0]['thumbnail_url'].endswith('/A/in.jpg'))
        with patch.object(self.db, 'get_all_records', side_effect=AssertionError('heavy records')):
            people = self.client.get('/api/people').json()['people']
        self.assertTrue(people[0]['thumbnail_url'].endswith('/A/in.jpg'))

    def test_old_route_removed_and_invalid_pagination(self):
        self.assertEqual(self.client.get('/api/visit-events').status_code, 404)
        self.assertNotIn('/api/visit-events', self.client.get('/openapi.json').json()['paths'])
        for query in ('limit=0', 'limit=-1', 'limit=1001', 'offset=-1'):
            self.assertEqual(self.client.get('/api/exits?' + query).status_code, 422)

    def test_websocket_initial_state_and_notification(self):
        with self.client.websocket_connect('/ws') as ws:
            initial = ws.receive_json()
            self.assertEqual(initial['data']['exits']['items'], [])
            self.assertNotIn('visit_events', initial['data'])
            person = self.person('A')
            self.event(person, 'entry', 10, 'in.jpg')
            self.event(person, 'exit', 20, 'out.jpg')
            response = self.client.post('/api/events', json=dict(event='outside', global_id=person, label='A', track_id=1, confidence=1, timestamp=20))
            self.assertEqual(response.status_code, 200)
            update = ws.receive_json()
            self.assertEqual(update['type'], 'outside')
            self.assertEqual(update['data']['exits']['total'], 1)


if __name__ == '__main__':
    unittest.main()
