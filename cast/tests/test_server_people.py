import json
import threading
import time
from urllib.request import Request, urlopen

from cast.server import CastHTTPServer
from cast.storage import TextFaceStore


def _post_json(url: str, payload: dict) -> dict:
    req = Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_update_person_endpoint(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    person = store.add_person(name="Caria (Friend of Lynda)")
    person_id = str(person["person_id"])

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        payload = _post_json(
            f"http://127.0.0.1:{port}/api/people/update",
            {"person_id": person_id, "display_name": "Carla (Friend of Lynda)"},
        )
        assert payload.get("ok") is True
        person_row = payload.get("person") or {}
        assert str(person_row.get("person_id")) == person_id
        assert str(person_row.get("display_name")) == "Carla (Friend of Lynda)"

        with urlopen(f"http://127.0.0.1:{port}/api/people", timeout=10) as response:
            people_payload = json.loads(response.read().decode("utf-8"))
        people = people_payload.get("people") or []
        updated = next(row for row in people if str(row.get("person_id")) == person_id)
        assert str(updated.get("display_name")) == "Carla (Friend of Lynda)"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
