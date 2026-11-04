from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from material_usage_perception.api.app import create_app
from material_usage_perception.sources import CameraSource, MappingCameraSourceRegistry


def test_camera_and_configuration_survive_restart_and_delete():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        app = create_app(storage_root=root)
        app.state.repository.add_configuration('cfg_test', {
            'configurationId': 'cfg_test', 'cameraId': 'persist_test', 'status': 'ACTIVE',
        })
        with TestClient(app) as client:
            assert client.post('/api/material-perception/v1/cameras', json={
                'cameraId': 'persist_test', 'streamUrl': 'rtsp://example.test/stream',
            }).status_code == 200
        with TestClient(create_app(storage_root=root)) as client:
            assert client.get('/files/data/.private/cameras.json').status_code == 404
            cameras = client.get('/api/material-perception/v1/cameras').json()['data']['items']
            assert any(c['cameraId'] == 'persist_test' for c in cameras)
            assert all('streamUrl' not in c for c in cameras)
            configs = client.get('/api/material-perception/v1/configurations').json()['data']['items']
            assert any(c['configurationId'] == 'cfg_test' for c in configs)
            assert client.delete('/api/material-perception/v1/cameras/persist_test').status_code == 200
            assert client.delete('/api/material-perception/v1/configurations/cfg_test').status_code == 200
        with TestClient(create_app(storage_root=root)) as client:
            cameras = client.get('/api/material-perception/v1/cameras').json()['data']['items']
            assert not any(c['cameraId'] == 'persist_test' for c in cameras)
            assert client.get('/api/material-perception/v1/configurations').json()['data']['items'] == []
            assert client.get('/api/material-perception/v1/configurations/cfg_test').status_code == 404
            assert client.delete('/api/material-perception/v1/cameras/persist_test').status_code == 404


def test_deleted_yaml_camera_does_not_return_and_can_be_registered_again():
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'cameras.json'
        source = CameraSource('yaml_camera', 'rtsp://example.test/stream')
        registry = MappingCameraSourceRegistry({source.camera_id: source})
        registry.enable_persistence(path)
        assert registry.delete(source.camera_id)
        restored = MappingCameraSourceRegistry({source.camera_id: source})
        restored.enable_persistence(path)
        assert restored.resolve(source.camera_id) is None
        restored.register(source)
        restarted = MappingCameraSourceRegistry()
        restarted.enable_persistence(path)
        assert restarted.resolve(source.camera_id) == source
