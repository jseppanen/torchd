import pytest
import requests
import subprocess
import torch
import time


@pytest.fixture
def torchd(tmpdir):
    torchd_path = "build/torchd"

    class ServerFixture:
        def start(self, model, *, device="cpu", host="127.0.0.1", port=7777):
            model_path = f"{tmpdir}/model.pt"
            script_model = torch.jit.script(model)
            torch.jit.save(script_model, model_path)
            cmd = [
                torchd_path,
                "--model",
                model_path,
                "--device",
                device,
                "--host",
                host,
                "--port",
                str(port),
            ]
            self.proc = subprocess.Popen(cmd, text=True)
            self.host = host
            self.port = port
            for _ in range(20):
                try:
                    self.ping()
                    return
                except requests.ConnectionError:
                    time.sleep(0.01)
            raise RuntimeError("server failed to start")

        def forward(self, *inputs):
            req = {"inputs": [torch.tensor(x).tolist() for x in inputs]}
            resp = requests.post(f"http://{self.host}:{self.port}/forward", json=req)
            if resp.status_code != 200:
                raise RuntimeError(f"forward failed with HTTP {resp.status_code}")
            json = resp.json()
            return json["output"]

        def ping(self):
            resp = requests.get(f"http://{self.host}:{self.port}/ping")
            if resp.status_code != 200:
                raise RuntimeError(f"ping failed with HTTP {resp.status_code}")

    server = ServerFixture()
    try:
        yield server
    finally:
        server.proc.terminate()
