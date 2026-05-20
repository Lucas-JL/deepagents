"""Unit tests for OSSBackend.

The oss2 SDK is mocked because unit tests may not make network calls.
"""

import base64
import json
from unittest.mock import MagicMock, patch

import oss2
import pytest

from deepagents.backends.oss_backend import OSSBackend
from deepagents.backends.protocol import (
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import create_file_data


@pytest.fixture
def backend():
    """Return an OSSBackend with a mocked bucket."""
    with patch.object(oss2, "Auth"), patch.object(oss2, "Bucket"):
        return OSSBackend(
            endpoint="https://oss-cn-hangzhou.aliyuncs.com",
            access_key_id="fake-id",
            access_key_secret="fake-secret",  # noqa: S106
            bucket_name="test-bucket",
            prefix="files/",
        )


@pytest.fixture
def backend_no_prefix():
    """Return an OSSBackend without a prefix."""
    with patch.object(oss2, "Auth"), patch.object(oss2, "Bucket"):
        return OSSBackend(
            endpoint="https://oss-cn-hangzhou.aliyuncs.com",
            access_key_id="fake-id",
            access_key_secret="fake-secret",  # noqa: S106
            bucket_name="test-bucket",
        )


class TestPathHelpers:
    def test_to_oss_key_with_prefix(self, backend):
        assert backend._to_oss_key("/hello.txt") == "files/hello.txt"
        assert backend._to_oss_key("/dir/file.py") == "files/dir/file.py"

    def test_to_oss_key_no_prefix(self, backend_no_prefix):
        assert backend_no_prefix._to_oss_key("/hello.txt") == "hello.txt"

    def test_to_virtual_path_with_prefix(self, backend):
        assert backend._to_virtual_path("files/hello.txt") == "/hello.txt"

    def test_to_virtual_path_no_prefix(self, backend_no_prefix):
        assert backend_no_prefix._to_virtual_path("hello.txt") == "/hello.txt"

    def test_roundtrip(self, backend):
        vpath = "/some/nested/file.md"
        assert backend._to_virtual_path(backend._to_oss_key(vpath)) == vpath


class TestWrite:
    def test_write_new_file(self, backend):
        backend._bucket.get_object = MagicMock(side_effect=type(backend)._get_file_data)

        backend._get_file_data = MagicMock(return_value=None)
        backend._put_file_data = MagicMock()

        result = backend.write("/new.txt", "hello world")
        assert isinstance(result, WriteResult)
        assert result.error is None
        assert result.path == "/new.txt"
        backend._put_file_data.assert_called_once()

    def test_write_existing_file_errors(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "old",
                "encoding": "utf-8",
            }
        )

        result = backend.write("/existing.txt", "new content")
        assert isinstance(result, WriteResult)
        assert result.error is not None
        assert "already exists" in result.error


class TestRead:
    def test_read_existing_file(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "line1\nline2\nline3",
                "encoding": "utf-8",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
            }
        )

        result = backend.read("/file.txt")
        assert isinstance(result, ReadResult)
        assert result.error is None
        assert result.file_data is not None
        assert "line1" in result.file_data["content"]

    def test_read_nonexistent_file(self, backend):
        backend._get_file_data = MagicMock(return_value=None)

        result = backend.read("/missing.txt")
        assert isinstance(result, ReadResult)
        assert result.error is not None
        assert "not found" in result.error

    def test_read_with_offset(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "line0\nline1\nline2\nline3\n",
                "encoding": "utf-8",
            }
        )

        result = backend.read("/file.txt", offset=2, limit=1)
        assert isinstance(result, ReadResult)
        assert result.error is None
        assert result.file_data is not None
        assert "line2" in result.file_data["content"]
        assert "line0" not in result.file_data["content"]


class TestEdit:
    def test_edit_replaces_string(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "hello world",
                "encoding": "utf-8",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
            }
        )
        backend._put_file_data = MagicMock()

        result = backend.edit("/file.txt", "world", "earth")
        assert isinstance(result, EditResult)
        assert result.error is None
        assert result.path == "/file.txt"
        assert result.occurrences == 1

        call_args = backend._put_file_data.call_args
        written_data = call_args[0][1]
        assert "earth" in written_data["content"]

    def test_edit_nonexistent_file(self, backend):
        backend._get_file_data = MagicMock(return_value=None)

        result = backend.edit("/missing.txt", "a", "b")
        assert isinstance(result, EditResult)
        assert result.error is not None
        assert "not found" in result.error

    def test_edit_no_match(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "hello world",
                "encoding": "utf-8",
            }
        )

        result = backend.edit("/file.txt", "xyz", "abc")
        assert isinstance(result, EditResult)
        assert result.error is not None

    def test_edit_replace_all(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "aaa bbb aaa",
                "encoding": "utf-8",
                "created_at": "2024-01-01T00:00:00",
            }
        )
        backend._put_file_data = MagicMock()

        result = backend.edit("/file.txt", "aaa", "ccc", replace_all=True)
        assert isinstance(result, EditResult)
        assert result.error is None
        assert result.occurrences == 2


class TestLs:
    def test_ls_lists_entries(self, backend):
        file_obj = MagicMock()
        file_obj.is_prefix.return_value = False
        file_obj.key = "files/hello.txt"
        file_obj.size = 42

        dir_obj = MagicMock()
        dir_obj.is_prefix.return_value = True
        dir_obj.key = "files/subdir/"

        with patch.object(oss2, "ObjectIterator", return_value=[file_obj, dir_obj]):
            result = backend.ls("/")
            assert result.error is None
            assert result.entries is not None
            assert len(result.entries) == 2

            paths = {e["path"] for e in result.entries}
            assert "/hello.txt" in paths
            assert "/subdir/" in paths


class TestGrepAndGlob:
    def test_grep_finds_matches(self, backend):
        backend._load_all_files = MagicMock(
            return_value={
                "/a.py": {"content": "import os\nprint('hello')", "encoding": "utf-8"},
                "/b.txt": {"content": "no match here", "encoding": "utf-8"},
            }
        )

        result = backend.grep("hello")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "/a.py"

    def test_grep_no_matches(self, backend):
        backend._load_all_files = MagicMock(
            return_value={
                "/a.txt": {"content": "nothing relevant", "encoding": "utf-8"},
            }
        )

        result = backend.grep("xyz")
        assert result.matches is not None
        assert len(result.matches) == 0

    def test_glob_matches_pattern(self, backend):
        backend._load_all_files = MagicMock(
            return_value={
                "/src/main.py": {
                    "content": "code",
                    "encoding": "utf-8",
                    "modified_at": "2024-01-01",
                },
                "/readme.md": {
                    "content": "docs",
                    "encoding": "utf-8",
                    "modified_at": "2024-01-02",
                },
            }
        )

        result = backend.glob("**/*.py", path="/")
        assert result.error is None
        assert result.matches is not None
        py_paths = [m["path"] for m in result.matches]
        assert "/src/main.py" in py_paths


class TestUploadDownload:
    def test_upload_new_file(self, backend):
        backend._get_file_data = MagicMock(return_value=None)
        backend._put_file_data = MagicMock()

        responses = backend.upload_files([("/new.txt", b"hello")])
        assert len(responses) == 1
        assert isinstance(responses[0], FileUploadResponse)
        assert responses[0].error is None
        backend._put_file_data.assert_called_once()

    def test_upload_binary_file(self, backend):
        backend._get_file_data = MagicMock(return_value=None)
        backend._put_file_data = MagicMock()

        binary_content = bytes(range(256))
        responses = backend.upload_files([("/bin.dat", binary_content)])
        assert responses[0].error is None

        call_args = backend._put_file_data.call_args
        written_data = call_args[0][1]
        assert written_data["encoding"] == "base64"

    def test_upload_preserves_encoding_on_update(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "old text",
                "encoding": "utf-8",
                "created_at": "2024-01-01T00:00:00",
            }
        )
        backend._put_file_data = MagicMock()

        responses = backend.upload_files([("/file.txt", b"new text")])
        assert responses[0].error is None

        call_args = backend._put_file_data.call_args
        written_data = call_args[0][1]
        assert written_data["encoding"] == "utf-8"

    def test_upload_encoding_change_creates_new(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "old text",
                "encoding": "utf-8",
                "created_at": "2024-01-01T00:00:00",
            }
        )
        backend._put_file_data = MagicMock()

        binary_content = bytes(range(256))
        responses = backend.upload_files([("/file.dat", binary_content)])
        assert responses[0].error is None

        call_args = backend._put_file_data.call_args
        written_data = call_args[0][1]
        assert written_data["encoding"] == "base64"

    def test_download_existing_file(self, backend):
        backend._get_file_data = MagicMock(
            return_value={
                "content": "hello",
                "encoding": "utf-8",
            }
        )

        responses = backend.download_files(["/file.txt"])
        assert len(responses) == 1
        assert isinstance(responses[0], FileDownloadResponse)
        assert responses[0].error is None
        assert responses[0].content == b"hello"

    def test_download_binary_file(self, backend):
        original = bytes(range(256))
        encoded = base64.b64encode(original).decode("ascii")

        backend._get_file_data = MagicMock(
            return_value={
                "content": encoded,
                "encoding": "base64",
            }
        )

        responses = backend.download_files(["/bin.dat"])
        assert responses[0].error is None
        assert responses[0].content == original

    def test_download_missing_file(self, backend):
        backend._get_file_data = MagicMock(return_value=None)

        responses = backend.download_files(["/missing.txt"])
        assert responses[0].error == "file_not_found"
        assert responses[0].content is None


class TestGetFileData:
    def test_get_file_data_success(self, backend):
        file_data = {
            "content": "hello",
            "encoding": "utf-8",
            "created_at": "2024-01-01T00:00:00",
            "modified_at": "2024-01-02T00:00:00",
        }
        mock_result = MagicMock()
        mock_result.read.return_value = json.dumps(file_data).encode("utf-8")
        backend._bucket.get_object = MagicMock(return_value=mock_result)

        result = backend._get_file_data("/test.txt")
        assert result is not None
        assert result["content"] == "hello"
        assert result["encoding"] == "utf-8"

    def test_get_file_data_not_found(self, backend):
        backend._bucket.get_object = MagicMock(
            side_effect=oss2.exceptions.NoSuchKey(404, {}, b"", {}),
        )

        result = backend._get_file_data("/missing.txt")
        assert result is None

    def test_get_file_data_oss_error(self, backend):
        backend._bucket.get_object = MagicMock(
            side_effect=oss2.exceptions.ServerError(500, {}, b"", {}),
        )

        result = backend._get_file_data("/error.txt")
        assert result is None


class TestPutFileData:
    def test_put_file_data_uploads_json(self, backend):
        backend._bucket.put_object = MagicMock()

        fd = create_file_data("test content")
        backend._put_file_data("/out.txt", fd)

        backend._bucket.put_object.assert_called_once()
        call_args = backend._bucket.put_object.call_args
        key = call_args[0][0]
        payload = call_args[0][1]

        assert key == "files/out.txt"
        parsed = json.loads(payload)
        assert parsed["content"] == "test content"
        assert parsed["encoding"] == "utf-8"
