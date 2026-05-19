"""OSSBackend: Alibaba Cloud OSS object storage backend."""

import base64
import json
import logging
from typing import Any

try:
    import oss2  # ty: ignore[unresolved-import]
except ImportError as _exc:
    msg = 'Could not import oss2. Please install it with: `pip install "deepagents[oss]"` or `pip install oss2`'
    raise ImportError(msg) from _exc

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    _get_file_type,
    _glob_search_files,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
    perform_string_replacement,
    slice_read_response,
    update_file_data,
)

logger = logging.getLogger(__name__)


class OSSBackend(BackendProtocol):
    """Backend that stores files in Alibaba Cloud OSS.

    Maps virtual paths (e.g. `/file.txt`) to OSS object keys
    (e.g. `prefix/file.txt`).  File content is persisted as
    JSON-serialized `FileData` objects.

    Example::

        from deepagents.backends import OSSBackend

        backend = OSSBackend(
            endpoint="https://oss-cn-hangzhou.aliyuncs.com",
            access_key_id="...",
            access_key_secret="...",
            bucket_name="my-bucket",
            prefix="agent-files/",
        )
    """

    def __init__(
        self,
        endpoint: str,
        access_key_id: str,
        access_key_secret: str,
        bucket_name: str,
        prefix: str = "",
    ) -> None:
        """Initialize OSSBackend.

        Args:
            endpoint: OSS endpoint URL (e.g. `https://oss-cn-hangzhou.aliyuncs.com`).
            access_key_id: Alibaba Cloud access key ID.
            access_key_secret: Alibaba Cloud access key secret.
            bucket_name: Name of the OSS bucket.
            prefix: Optional key prefix for all objects (e.g. `"agent-files/"`).
        """
        auth = oss2.Auth(access_key_id, access_key_secret)
        self._bucket = oss2.Bucket(auth, endpoint, bucket_name)
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _to_oss_key(self, virtual_path: str) -> str:
        """Convert a virtual path to an OSS object key.

        Strips the leading `/` and prepends the configured prefix.
        """
        stripped = virtual_path.lstrip("/")
        return f"{self._prefix}{stripped}"

    def _to_virtual_path(self, oss_key: str) -> str:
        """Convert an OSS object key back to a virtual path."""
        relative = oss_key
        if self._prefix and oss_key.startswith(self._prefix):
            relative = oss_key[len(self._prefix) :]
        return f"/{relative}"

    # ------------------------------------------------------------------
    # Internal I/O helpers
    # ------------------------------------------------------------------

    def _put_file_data(self, virtual_path: str, file_data: FileData) -> None:
        """Serialize `file_data` as JSON and upload to OSS."""
        payload = json.dumps(dict(file_data), ensure_ascii=False).encode("utf-8")
        key = self._to_oss_key(virtual_path)
        self._bucket.put_object(key, payload)

    def _get_file_data(self, virtual_path: str) -> FileData | None:
        """Download an object from OSS and deserialize to `FileData`.

        Returns `None` when the object does not exist.
        """
        key = self._to_oss_key(virtual_path)
        try:
            result = self._bucket.get_object(key)
            raw = result.read()
            data: dict[str, Any] = json.loads(raw)
            return FileData(
                content=data["content"],
                encoding=data.get("encoding", "utf-8"),
                **{k: data[k] for k in ("created_at", "modified_at") if k in data},
            )
        except oss2.exceptions.NoSuchKey:
            return None
        except oss2.exceptions.OssError:
            logger.exception("OSS error reading %s", virtual_path)
            return None

    def _list_keys(self, prefix: str = "") -> list[str]:
        """List all object keys under `prefix` (non-recursive listing helper)."""
        oss_prefix = self._to_oss_key(prefix.lstrip("/")) if prefix else self._prefix
        return [obj.key for obj in oss2.ObjectIterator(self._bucket, prefix=oss_prefix)]

    # ------------------------------------------------------------------
    # BackendProtocol implementation
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        """List files and directories in the specified directory.

        Args:
            path: Virtual directory path.

        Returns:
            `LsResult` with entries.
        """
        oss_prefix = self._to_oss_key(path.lstrip("/"))
        if oss_prefix and not oss_prefix.endswith("/"):
            oss_prefix += "/"

        entries: list[FileInfo] = []
        seen_dirs: set[str] = set()

        for obj in oss2.ObjectIterator(self._bucket, prefix=oss_prefix, delimiter="/"):
            if obj.is_prefix():
                dir_path = self._to_virtual_path(obj.key)
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    entries.append(FileInfo(path=dir_path, is_dir=True, size=0))
            else:
                if obj.key == oss_prefix:
                    continue
                vpath = self._to_virtual_path(obj.key)
                entries.append(
                    FileInfo(
                        path=vpath,
                        is_dir=False,
                        size=int(obj.size),
                    )
                )

        entries.sort(key=lambda e: e.get("path", ""))
        return LsResult(entries=entries)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content for the requested line range.

        Args:
            file_path: Virtual file path.
            offset: Line offset (0-indexed).
            limit: Maximum number of lines.

        Returns:
            `ReadResult` with file data or error.
        """
        file_data = self._get_file_data(file_path)
        if file_data is None:
            return ReadResult(error=f"File '{file_path}' not found")

        if _get_file_type(file_path) != "text":
            return ReadResult(file_data=file_data)

        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced

        sliced_fd = FileData(
            content=sliced,
            encoding=file_data.get("encoding", "utf-8"),
        )
        if "created_at" in file_data:
            sliced_fd["created_at"] = file_data["created_at"]
        if "modified_at" in file_data:
            sliced_fd["modified_at"] = file_data["modified_at"]
        return ReadResult(file_data=sliced_fd)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Write content to a new file, error if file already exists.

        Args:
            file_path: Virtual file path.
            content: String content to write.

        Returns:
            `WriteResult` with path or error.
        """
        existing = self._get_file_data(file_path)
        if existing is not None:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.",
            )

        new_file_data = create_file_data(content)
        self._put_file_data(file_path, new_file_data)
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Virtual file path.
            old_string: Text to find.
            new_string: Replacement text.
            replace_all: Replace all occurrences when `True`.

        Returns:
            `EditResult` with path and occurrence count, or error.
        """
        file_data = self._get_file_data(file_path)
        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        self._put_file_data(file_path, new_file_data)
        return EditResult(path=file_path, occurrences=int(occurrences))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search OSS files for a literal text pattern.

        Downloads matching files and searches their content.

        Args:
            pattern: Literal string to search for.
            path: Optional directory to restrict the search.
            glob: Optional filename glob filter.

        Returns:
            `GrepResult` with matches.
        """
        files = self._load_all_files(path)
        return grep_matches_from_files(files, pattern, path if path is not None else "/", glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g. `"*.py"`).
            path: Base directory to search from.

        Returns:
            `GlobResult` with matching file info.
        """
        files = self._load_all_files(path)
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return GlobResult(matches=[])

        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            size = len(fd.get("content", "")) if fd else 0
            infos.append(
                FileInfo(
                    path=p,
                    is_dir=False,
                    size=int(size),
                    modified_at=fd.get("modified_at", "") if fd else "",
                )
            )
        return GlobResult(matches=infos)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to OSS.

        Args:
            files: List of `(path, content_bytes)` tuples.

        Returns:
            List of `FileUploadResponse` objects.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                text = content.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                text = base64.b64encode(content).decode("ascii")
                encoding = "base64"

            existing = self._get_file_data(path)
            file_data = (
                update_file_data(existing, text)
                if existing and existing.get("encoding", "utf-8") == encoding
                else create_file_data(text, encoding=encoding)
            )
            self._put_file_data(path, file_data)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from OSS.

        Args:
            paths: List of virtual file paths.

        Returns:
            List of `FileDownloadResponse` objects.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            file_data = self._get_file_data(path)
            if file_data is None:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="file_not_found"),
                )
                continue

            content_str = file_data_to_string(file_data)
            encoding = file_data.get("encoding", "utf-8")
            content_bytes = content_str.encode("utf-8") if encoding == "utf-8" else base64.standard_b64decode(content_str)
            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))
        return responses

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all_files(self, path: str | None = None) -> dict[str, Any]:
        """Load all `FileData` objects under `path` into a dict.

        Used by `grep` and `glob` which need in-memory access to file
        contents for pattern matching.
        """
        search_path = path if path is not None else "/"
        keys = self._list_keys(search_path)

        files: dict[str, Any] = {}
        for key in keys:
            if key.endswith("/"):
                continue
            vpath = self._to_virtual_path(key)
            file_data = self._get_file_data(vpath)
            if file_data is not None:
                files[vpath] = file_data
        return files
