"""Shared, resumable downloads for the Linux/macOS data preparation pipeline."""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


@contextmanager
def data_lock(path):
    """Prevent concurrent ranks/workers from preparing the same files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def nonempty(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def _valid(path, size, md5):
    if not nonempty(path):
        return False
    if size is not None and path.stat().st_size != size:
        return False
    if md5:
        digest = hashlib.md5()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest() == md5
    return True


def download_file(url, destination, *, size=None, md5=None, retries=3):
    """Stream to .part, resume HTTP ranges, and publish only complete files."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + '.part')
    with data_lock(str(destination) + '.lock'):
        if _valid(destination, size, md5):
            return destination
        for attempt in range(retries):
            try:
                if size and nonempty(partial) and partial.stat().st_size == size:
                    if _valid(partial, size, md5):
                        os.replace(partial, destination)
                        return destination
                    partial.unlink()
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {'User-Agent': 'ReSFlow-data/1.0', 'Accept-Encoding': 'identity'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                request = urllib.request.Request(url, headers=headers)
                print(f'[data] Downloading {destination.name}'
                      + (f' (resuming at {offset:,} bytes)' if offset else ''), flush=True)
                with urllib.request.urlopen(request, timeout=60) as response:
                    status = getattr(response, 'status', 200)
                    if status == 206:
                        content_range = response.headers.get('Content-Range', '')
                        if not content_range.startswith(f'bytes {offset}-'):
                            raise ValueError(f'Unexpected Content-Range: {content_range}')
                        total = int(content_range.rsplit('/', 1)[1])
                    else:
                        offset = 0  # Server ignored Range: start over, never append.
                        length = response.headers.get('Content-Length')
                        total = int(length) if length else None
                    mode = 'ab' if offset else 'wb'
                    written = offset
                    next_report = written + 256 * 1024 * 1024
                    with partial.open(mode) as handle:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            written += len(chunk)
                            if written >= next_report:
                                suffix = f' / {total / 1e9:.2f} GB' if total else ''
                                print(f'[data] {destination.name}: {written / 1e9:.2f} GB{suffix}', flush=True)
                                next_report = written + 256 * 1024 * 1024
                    expected = size if size is not None else total
                    if not _valid(partial, expected, md5):
                        # A complete transfer with a bad checksum must restart.
                        if expected is not None and written >= expected:
                            partial.unlink(missing_ok=True)
                        raise ValueError(f'Incomplete download or checksum mismatch: {destination.name}')
                os.replace(partial, destination)
                return destination
            except (OSError, ValueError, urllib.error.URLError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code == 416:
                    partial.unlink(missing_ok=True)
                if attempt + 1 == retries:
                    raise RuntimeError(f'Could not download {url} to {destination}: {error}') from error
                time.sleep(attempt + 1)
