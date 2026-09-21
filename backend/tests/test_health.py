from fastapi.testclient import TestClient

from voicelm import __version__
from voicelm.api.app import create_app


def test_health_reports_ok(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path / "data")) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": __version__}


def test_unknown_route_is_404(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path / "data")) as client:
        assert client.get("/does-not-exist").status_code == 404
