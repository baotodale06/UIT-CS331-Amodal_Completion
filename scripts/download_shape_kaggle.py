"""Download one private Kaggle version; keep signed download URLs out of logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import json
import os

from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
import requests


def download(api, kernel, destination):
    owner, slug, version = api.parse_kernel_string(kernel)
    if not version:
        raise ValueError('An explicit kernel version is required')
    destination.mkdir(parents=True, exist_ok=True)
    items, token, log = [], None, None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.version_label = api._kernel_version_label(version)
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            items.extend(response.files or [])
            log = log or response.log
            token = response.next_page_token
            if not token:
                break
    names = [item.file_name for item in items]
    if len(names) != len(set(names)):
        raise ValueError('Repeated output paths')
    for name in names:
        if '\\' in name or ':' in name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Unsafe output path')
        if not (destination/name).resolve().is_relative_to(destination.resolve()):
            raise ValueError('Output path escaped destination')

    def fetch(item):
        path = destination/item.file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.download-part')
        try:
            with requests.get(item.url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                expected = response.headers.get('Content-Length')
                if path.exists():
                    if expected is not None and path.stat().st_size == int(expected):
                        return
                    raise ValueError('Preserve differing recovered output')
                with temporary.open('wb') as stream:
                    for data in response.iter_content(1024*1024):
                        stream.write(data)
                if expected is not None and temporary.stat().st_size != int(expected):
                    raise ValueError('Incomplete download')
                os.replace(temporary, path)
        except Exception as error:
            raise RuntimeError(f'Download failed for {item.file_name}: {type(error).__name__}') from None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, _ in enumerate(pool.map(fetch, items), 1):
            if count % 100 == 0:
                print(json.dumps(dict(downloaded=count, files=len(items))), flush=True)
    if log:
        log_path = destination/(slug+'.log')
        if log_path.exists() and log_path.read_text(encoding='utf8') != log:
            raise ValueError('Preserve differing version log')
        log_path.write_text(log, encoding='utf8')
    print(json.dumps(dict(kernel=kernel, files=len(items), downloaded=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    download(api, args.kernel, args.output)


if __name__ == '__main__':
    main()
