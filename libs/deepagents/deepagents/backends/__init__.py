"""Memory backends for pluggable file storage."""

import contextlib

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.context_hub import ContextHubBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.langsmith import LangSmithSandbox
from deepagents.backends.local_shell import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend
from deepagents.backends.protocol import BackendProtocol

with contextlib.suppress(ImportError):
    from deepagents.backends.oss_backend import OSSBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import (
    BackendContext,
    NamespaceFactory,
    StoreBackend,
)

__all__ = [
    "DEFAULT_EXECUTE_TIMEOUT",
    "BackendContext",
    "BackendProtocol",
    "CompositeBackend",
    "ContextHubBackend",
    "FilesystemBackend",
    "LangSmithSandbox",
    "LocalShellBackend",
    "NamespaceFactory",
    "OSSBackend",
    "StateBackend",
    "StoreBackend",
]
