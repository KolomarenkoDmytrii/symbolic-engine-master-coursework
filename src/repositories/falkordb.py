from typing import Any

import retrico


class FalkorDBGraphRepository:
    def __init__(self, config: Any) -> None:
        self._store = retrico.create_graph_store(config)

    def __getattr__(self, name: str) -> Any:
        # Forward methods such as get_entity_by_id(), add_entity(),
        # add_relation(), and other methods provided by the Retrico store.
        return getattr(self._store, name)

    def close(self) -> Any:
        close = getattr(self._store, "close", None)
        if callable(close):
            return close()