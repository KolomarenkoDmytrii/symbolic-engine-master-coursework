```sh
devpod provider add docker \
  -o DOCKER_PATH="$(command -v podman)" \
  -o DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/podman/podman.sock"

devpod provider use docker
```

```
pylint --disable=missing-module-docstring,missing-class-docstring,missing-function-docstring src
```

```
mypy src
```

```
uvicorn src.main:app --host 0.0.0.0 --port 8000
```