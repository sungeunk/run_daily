try:
    import aiohttp
    import argparse
    import asyncio
    import datetime
    import logging as log
    import os
    import platform
    import re
    import requests
    import shutil
    import subprocess
    import sys
    import tarfile
    import tqdm.asyncio as tqdm_asyncio
    import zipfile
    import yaml

    from bs4 import BeautifulSoup
    from functools import total_ordering
    from pathlib import Path  # Use pathlib for path manipulation
    from tabulate import tabulate
    from tqdm import tqdm
except ImportError:
    # Exit if critical dependencies are missing
    print(f'Please install these modules: pip install aiohttp requests tqdm pyyaml bs4 python-dateutil gitpython packaging tabulate')
    sys.exit(-1)



################################################
# Global variable
################################################

# Check if the operating system is Windows
IS_WINDOWS = platform.system() == 'Windows'
# Get the current working directory as a Path object
PWD = Path.cwd()
STORAGE_OV_FILETREE_JSON = 'https://storage.openvinotoolkit.org/filetree.json'


# Attempt to find the Ubuntu version if on Linux
UBUNTU_VER = ''
if platform.system() == 'Linux':
    try:
        # Run lsb_release to get distribution info
        output = subprocess.check_output(['lsb_release', '-r'], text=True)
        match_obj = re.search(r'Release:[ \t]+(\d+).(\d+)', output)
        if match_obj:
            UBUNTU_VER = f'{match_obj.groups()[0]}_{match_obj.groups()[1]}'
    except FileNotFoundError:
        log.warning('lsb_release not found. Assuming non-Ubuntu Linux.')
    except Exception as e:
        log.warning(f'Could not determine Ubuntu version: {e}')

# --- Utility Functions ---

def check_filepath(path: Path) -> bool:
    """Checks if a file or directory exists at the given Path."""
    try:
        # Ensure the input is a Path object before calling .exists()
        if not isinstance(path, Path):
            path = Path(path)
        return path.exists()
    except Exception as e:
        # Catch potential errors like permission issues
        log.error(f'could not check existence of: {path} ({e})')
    return False

def get_text_from_web(url: str) -> str:
    """Fetches text content from a URL."""
    try:
        # [User Request] Keep SSL verification disabled
        res = requests.get(url, verify=False)
        if res.ok:
            return res.text
        else:
            log.error(f'Could not get file: res({res.status_code}/{res.reason}) {url}')
            return ''
    except requests.exceptions.RequestException as e:
        log.error(f'get_text_from_web::RequestException: {e}')
        return ''

def download_file(url: str, out_path: Path) -> Path | None:
    """
    Downloads a single file from a URL to a specified directory with a TQDM progress bar.

    Args:
        url: The URL of the file to download.
        out_path: The directory (Path object) to save the file in.

    Returns:
        The Path to the downloaded file, or None on failure.
    """
    try:
        # [User Request] Keep SSL verification disabled
        res = requests.get(url, stream=True, verify=False)
        if res.ok:
            # Ensure the output directory exists
            out_path.mkdir(parents=True, exist_ok=True)

            total = int(res.headers.get('content-length', 0))
            filename = Path(url).name
            filepath = out_path / filename

            # Use synchronous tqdm for this synchronous download
            with open(filepath, 'wb') as file, tqdm(
                desc=filename,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in res.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)  # Update progress bar by bytes written
            return filepath.resolve()
        else:
            log.error(f'Could not get file: res({res.status_code}/{res.reason}) {url}')
    except requests.exceptions.RequestException as e:
        log.error(f'download_file::RequestException: {e}')
    return None

async def async_download_files(url_list: list, out_path: Path) -> list:
    """
    Downloads a list of files concurrently using aiohttp and asyncio.

    Args:
        url_list: A list of string URLs to download.
        out_path: The directory (Path object) to save files in.

    Returns:
        A list of Paths to the downloaded files.
    """
    # Limit concurrent downloads to 5 to avoid overwhelming the server or network
    semaphore = asyncio.Semaphore(5)

    async def async_download_file(session, url, out_path):
        """Inner function to download a single file within the async context."""
        async with semaphore:
            try:
                # [User Request] Keep SSL verification disabled
                async with session.get(url, verify_ssl=False) as res:
                    if res.ok:
                        filename = Path(url).name
                        filepath = out_path / filename

                        with open(filepath, 'wb') as file:
                            async for data in res.content.iter_chunked(1024*4):
                                file.write(data)
                            return filepath.resolve()
                    else:
                        raise Exception(f'Could not get file: res({res.status_code}/{res.reason}) {url}')
            except Exception as e:
                log.error(f'Failed to download {url}: {e}')
                return None

    async with aiohttp.ClientSession() as session:
        tasks = [async_download_file(session, url, out_path) for url in url_list]
        # Use tqdm_asyncio.tqdm.gather to show progress for async tasks
        results = await tqdm_asyncio.tqdm.gather(*tasks)
        # Filter out any None results from failed downloads
        return [r for r in results if r]

# --- OpenVINO Specific Functions ---

def required_openvino_packages_list() -> list:
    """Returns a static list of required OpenVINO package filenames."""
    REQUIRED_LIST = [
        'benchmark_app.zip', 'core.zip', 'core_c.zip', 'core_c_dev.zip',
        'core_dev.zip', 'cpp_samples.zip', 'cpu.zip', 'gpu.zip', 'ir.zip',
        'onnx.zip', 'openvino_req_files.zip', 'ovc.zip', 'paddle.zip',
        'pyopenvino_python3.10.zip', 'pyopenvino_python3.11.zip',
        'pyopenvino_python3.12.zip', 'pytorch.zip', 'setupvars.zip',
        'tbb.zip', 'tbb_dev.zip', 'tensorflow.zip', 'tensorflow_lite.zip',
    ]
    return REQUIRED_LIST

def required_genai_packages_list() -> list:
    """Returns a static list of required GenAI package filenames."""
    REQUIRED_LIST = [
        'openvino_tokenizers.zip', 'core_genai.zip', 'core_genai_dev.zip',
        'pygenai_3_10.zip', 'pygenai_3_11.zip', 'pygenai_3_12.zip',
    ]
    return REQUIRED_LIST

def check_required_packages(url: str) -> bool:
    """
    Checks if all required OV and GenAI packages are listed in the HTML content of the URL.
    """
    check_list = required_openvino_packages_list() + required_genai_packages_list()

    text = get_text_from_web(url)
    if not text:
        return False

    for item in check_list:
        # Use regex to find the filename in the text
        if re.search(item, text) is None:
            log.warning(f'Missing required package {item} at URL: {url}')
            return False
    return True

@total_ordering
class OV_VERSION:
    """
    A class to parse and compare OpenVINO-style version strings.
    Format: {year}.{major}.{minor}-{commit_number}-{commit_id}{optional_extra}
    Example: 2026.0.0-20441-4eb08a2a829
    Example: 2025.3.0-19807-44526285f24-releases/2025/3
    """
    # Regex: (year).(major).(minor)-(commit_num)-(commit_id)
    # Assumes commit_id consists only of alphanumeric characters (a-zA-Z0-9).
    VERSION_REGEX = re.compile(
        r"^(\d+)\.(\d+)\.(\d+)-(\d+)-([a-zA-Z0-9]+)"
    )

    def __init__(self, version_string: str):
        self.raw_string = version_string.strip()

        match = self.VERSION_REGEX.match(self.raw_string)
        if not match:
            raise ValueError(f"Invalid version string format: '{self.raw_string}'")
        groups = match.groups()

        # Store parsed components
        self.year = int(groups[0])
        self.major = int(groups[1])
        self.minor = int(groups[2])
        self.commit_number = int(groups[3])
        self.commit_id = groups[4]

        # Tuple used for magnitude comparison
        # Compares in the order: (year, major, minor, commit_number)
        self.version_tuple = (
            self.year,
            self.major,
            self.minor,
            self.commit_number
        )

    def __repr__(self) -> str:
        """
        Returns an unambiguous string representation of the object (for debugging).
        """
        return f"OV_VERSION('{self.raw_string}')"

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation (e.g., for print()).
        """
        return f'{self.year}.{self.major}.{self.minor}-{self.commit_number}-{self.commit_id}'

    def _check_other(self, other) -> tuple | type(NotImplemented):
        """
        Checks if 'other' is an OV_VERSION instance.
        If not, returns NotImplemented.
        """
        if not isinstance(other, OV_VERSION):
            return NotImplemented
        return other.version_tuple

    # --- Methods required for @total_ordering ---

    def __eq__(self, other) -> bool:
        """
        '==' (Equality) comparison.
        Returns True only if all parsed components match exactly.
        """
        if not isinstance(other, OV_VERSION):
            return NotImplemented
        return (self.version_tuple == other.version_tuple and
                self.commit_id == other.commit_id)

    def __lt__(self, other) -> bool:
        """
        '<' (Less than) comparison.
        Compares based on the version_tuple (year, major, minor, commit_number).
        """
        other_tuple = self._check_other(other)
        if other_tuple is NotImplemented:
            return NotImplemented
        return self.version_tuple < other_tuple

def get_ov_version(url: str) -> OV_VERSION | None:
    """Parses the OpenVINO version from the HTML content of a cpack URL."""
    manifest_url = generate_manifest_url_from_cpack(url)
    manifest_data = get_text_from_web(manifest_url)
    if manifest_data:
        data_dic = yaml.safe_load(manifest_data)
        return OV_VERSION(data_dic['components']['dldt']['version'])
    else:
        return None

def save_ov_version(args: argparse.Namespace, new_version: OV_VERSION):
    """Saves the given version string to the OV_NIGHTLY_VERSION cache file."""
    version_file = args.output / 'OV_NIGHTLY_VERSION'
    try:
        with open(version_file, 'w', encoding='utf8') as fos:
            fos.write(f'{new_version}')
    except IOError as e:
        log.error(f'Failed to write version file: {e}')

def load_local_ov_version(args: argparse.Namespace) -> OV_VERSION | None:
    """Loads the last saved version string from the OV_NIGHTLY_VERSION cache file."""
    version_file = args.output / 'OV_NIGHTLY_VERSION'
    try:
        with open(version_file, 'r', encoding='utf8') as fis:
            return OV_VERSION(fis.readline().strip())  # .strip() to remove newlines
    except FileNotFoundError:
        return None
    except IOError as e:
        return None

def check_ov_version(exist_ov_version: OV_VERSION, url: str) -> bool:
    """
    Compares the version at the URL with the locally saved version.

    Returns:
        True if the new version is newer, or if no old version is found.
        False if the new version is older or the same, or if parsing fails.
    """
    if exist_ov_version == None:
        return True
    target_version = get_ov_version(url)
    if target_version == None:
        return False

    # Return True only if the new version is strictly greater
    return exist_ov_version < target_version

def download_openvino_packages(url: str, out_path: Path) -> tuple[list, Path]:
    """
    Downloads all required OpenVINO packages from a specific cpack URL.

    Args:
        url: The root URL of the cpack directory.
        out_path: The base output directory (e.g., 'openvino_nightly').

    Returns:
        A tuple containing:
        (list of downloaded file Paths, Path to the new versioned output directory)
    """
    REQUIRED_LIST = required_openvino_packages_list()

    # Parse commit ID from URL
    COMMIT_ID = None
    match_obj = re.search(r'(custom_build|commit|pre_commit)\/([\w]+)', url)
    if match_obj:
        COMMIT_ID = match_obj.groups()[1]
    if not COMMIT_ID:
        raise Exception(f'could not parse COMMIT_ID url: {url}')

    text = get_text_from_web(url)
    log.debug(f'root: {text}')

    # Parse OV version from text
    if IS_WINDOWS:
        match_obj = re.search(fr'inference-engine_Release-(\d+.\d+.\d+.\d+)-win64-{REQUIRED_LIST[0]}', text)
    else:
        match_obj = re.search(fr'inference-engine-(\d+.\d+.\d+.\d+)-Linux-{REQUIRED_LIST[0]}', text)
    if not match_obj:
        raise Exception(f'could not parse the OV_VERSION from text: {text}')

    OV_VERSION = match_obj.groups()[0]
    # Create a version/commit-specific output directory
    new_out_path = out_path / f'{OV_VERSION}_{COMMIT_ID[0:8]}'
    new_out_path.mkdir(parents=True, exist_ok=True)

    # Build the full list of URLs to download
    if IS_WINDOWS:
        list_urls = [ f'{url}/inference-engine_Release-{OV_VERSION}-win64-{target_filename}' for target_filename in REQUIRED_LIST ]
    else:
        list_urls = [ f'{url}/inference-engine-{OV_VERSION}-Linux-{target_filename}' for target_filename in REQUIRED_LIST ]

    # Download all files concurrently
    downloaded_files = asyncio.run(async_download_files(list_urls, new_out_path))
    return downloaded_files, new_out_path

def download_genai_packages(url: str, ov_dst_path: Path) -> list:
    """Downloads all required GenAI packages to the *same* directory as the OV packages."""
    REQUIRED_LIST = required_genai_packages_list()

    text = get_text_from_web(url)
    os_name = 'win64' if IS_WINDOWS else 'Linux'
    # Example: OpenVINOGenAI-1.0.0-win64-openvino_tokenizers.zip
    match_obj = re.search(fr'OpenVINOGenAI-([\d.]+)-{os_name}-{REQUIRED_LIST[0]}', text)
    if not match_obj:
        raise Exception(f'could not parse the GENAI_VERSION from text: {text}')

    GENAI_VERSION = match_obj.groups()[0]
    list_urls = [ f'{url}/OpenVINOGenAI-{GENAI_VERSION}-{os_name}-{target_filename}' for target_filename in REQUIRED_LIST ]

    return asyncio.run(async_download_files(list_urls, ov_dst_path))

def decompress(compressed_filepath: Path, store_path: Path, delete_zip: bool = False) -> Path | None:
    """
    Decompresses a .zip or .tgz file into the store_path.
    This function no longer uses os.chdir().

    Args:
        compressed_filepath: Path to the .zip or .tgz file.
        store_path: Path to the directory where contents will be extracted.
        delete_zip: If True, delete the original archive after extraction.

    Returns:
        Path to the root of the uncompressed content, or None on failure.
    """
    root, ext = os.path.splitext(compressed_filepath.name)

    if ext == '.zip':
        if IS_WINDOWS:
            with zipfile.ZipFile(compressed_filepath, 'r') as file:
                file.extractall(store_path)
        else:
            try:
                command = ['unzip', '-o', '-q', compressed_filepath, '-d', store_path]
                subprocess.run(command, check=True, capture_output=True, text=True)
            except FileNotFoundError:
                print("Error: could not find 'unzip'", file=sys.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Error: 'unzip' execution failture (ReturnCode {e.returncode}):", file=sys.stderr)
                print(f"STDERR: {e.stderr}", file=sys.stderr)
    elif ext == '.tgz' or ext == '.gz':
        with tarfile.open(compressed_filepath, 'r:gz') as tar:
            tar.extractall(store_path, filter='tar', numeric_owner=True)
    else:
        log.warning(f'Unknown compression format: {ext}')
        return None

    if delete_zip:
        os.remove(compressed_filepath)

    # Return the path to the directory that was likely created
    # Note: This assumes the archive extracts to a dir named like the archive
    return store_path / root

def install_openvino(ov_filepath: Path, output_dir: Path):
    """Wrapper function to decompress a single OV package and update the setup path."""
    if not check_filepath(ov_filepath):
        log.warning(f'no file: {ov_filepath}')
        return

    uncompressed_dir = decompress(ov_filepath, output_dir)
    if uncompressed_dir:
        setup_script = uncompressed_dir / ('setupvars.bat' if IS_WINDOWS else 'setupvars.sh')
        update_latest_ov_setup_file(setup_script, output_dir)

def update_latest_ov_setup_file(setup_script_path: Path, output_dir: Path):
    """Saves the path to the latest setup script to 'latest_ov_setup_file.txt'."""
    latest_ov_setup_file = output_dir / 'latest_ov_setup_file.txt'

    try:
        with open(latest_ov_setup_file, 'w') as fos:
            fos.write(str(setup_script_path))  # Write the path
        log.info(f'New setup script: {setup_script_path}')
        log.info(f'Please setup with exist ov package: {latest_ov_setup_file}')
    except IOError as e:
        log.error(f'Failed to write latest setup file: {e}')

# --- Git & Build Server Functions ---

def get_child(json_obj, name):
    """Helper for traversing the filetree.json."""
    if "children" not in json_obj:
        return None
    for child_obj in json_obj["children"]:
        if child_obj is not None and child_obj.get("name") == name:
            return child_obj
    return None

def get_directory(json_obj, directory):
    """Helper for traversing the filetree.json."""
    dirs = directory.split('/')
    for dir_name in dirs:
        json_obj = get_child(json_obj, dir_name.strip())
        if json_obj is None:
            return None
    return json_obj

def get_private_cpack_url() -> str:
    """Returns the platform-specific cpack directory name."""
    if IS_WINDOWS:
        return 'private_windows_vs2022_release/cpack'
    else:
        # Use the detected UBUNTU_VER
        return f'private_linux_ubuntu_{UBUNTU_VER}_release/cpack'

def get_latest_commit_list_from_openvino() -> list:
    """
    Fetches the latest 10 commit hashes from the openvino master branch
    using the GitHub REST API instead of cloning.
    """
    # GitHub API endpoint for listing commits
    API_URL = 'https://api.github.com/repos/openvinotoolkit/openvino/commits'

    # --- GitHub API Authentication (Highly Recommended) ---
    # To avoid rate limiting (60 requests/hr for unauthenticated IPs),
    # use a Personal Access Token (PAT) read from an environment variable.
    token = os.environ.get('GITHUB_TOKEN')
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'X-GitHub-Api-Version': '2022-11-28' # Best practice
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'
        log.info('Using GitHub API with authentication token.')
    else:
        log.warning('GITHUB_TOKEN env var not set. Using unauthenticated GitHub API calls, which are rate-limited (60/hr).')
    # --- End Authentication ---

    # Parameters for the API request:
    # sha=master -> get commits from the 'master' branch
    # per_page=10 -> get only the 10 most recent commits
    params = {
        'sha': 'master',
        'per_page': 40
    }

    try:
        log.info(f'Fetching latest 10 commits from {API_URL}...')
        res = requests.get(API_URL, headers=headers, params=params, timeout=10)

        # Raise an exception for bad responses (403, 404, 500, etc.)
        res.raise_for_status()

        data = res.json()

        # Extract just the commit SHA hash from each commit object
        commit_list = [commit['sha'] for commit in data]

        if not commit_list:
            log.warning('GitHub API returned an empty commit list.')

        return commit_list

    except requests.exceptions.HTTPError as http_err:
        log.error(f'GitHub API HTTP error: {http_err}')
        if res.status_code == 403:
            log.error('API rate limit likely exceeded. Set the GITHUB_TOKEN environment variable.')
    except requests.exceptions.RequestException as e:
        log.error(f'Failed to get commits from GitHub API: {e}')

    return [] # Return an empty list on failure

def get_list_of_openvino_master(args: argparse.Namespace) -> tuple[list, bool]:
    """
    Scrapes the private build server for recent, valid master branch builds.

    Returns:
        A tuple containing:
        (list of valid cpack URLs, bool indicating if an old version exists)
    """
    MASTER_COMMIT_URL_ROOT = 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/commit'
    log.info(f'Querying openvino list: {MASTER_COMMIT_URL_ROOT}')

    root = get_text_from_web(MASTER_COMMIT_URL_ROOT)
    if not root:
        log.error('Failed to get master commit list HTML.')
        return [], False

    MASTER_COMMIT_URL_TEMPLATE = f'{MASTER_COMMIT_URL_ROOT}/COMMIT_ID/{get_private_cpack_url()}'

    master_commit_list = []

    # Use BeautifulSoup for reliable HTML parsing
    soup = BeautifulSoup(root, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        # Example: <a href="fa6b0ec1715d9f09ef17a9ac1a052e4c20781288/">...</a>
        if href and len(href) > 20 and href.endswith('/'):
            commit_id = href[:-1]  # Remove trailing slash
            # Get the date text, which is the next sibling
            date_str = link.next_sibling
            match = re.search(r'(\d{2}-[a-zA-Z]{3}-\d{4} \d{2}:\d{2})', str(date_str))
            if match:
                # We found the date string, e.g., "26-Sep-2025 14:34"
                date_string_cleaned = match.group(1)
                try:
                    dateobj = datetime.datetime.strptime(date_string_cleaned, '%d-%b-%Y %H:%M')
                    master_commit_list.append([commit_id, dateobj])
                except ValueError as e:
                    # This should rarely happen now, but good to keep
                    log.warning(f'strptime failed even after regex clean: {e}')
            else:
                # The text node didn't contain a date (e.g., "Parent Directory" link)
                continue

    # Sort by date, newest first
    master_commit_list.sort(key = lambda x : x[1], reverse=True)

    old_ov_version = load_local_ov_version(args)
    latest_commit_list = get_latest_commit_list_from_openvino()
    if not latest_commit_list:
        log.warning('Could not get latest commit list from GitHub, proceeding anyway...')

    exist_ov_version = load_local_ov_version(args)
    ret_list = []
    # Check the top 40 builds from the server
    for commit, date in master_commit_list[:40]:
        # If we have a GitHub commit list, filter by it
        if latest_commit_list and not commit in latest_commit_list:
            log.debug(f'{commit} is not in latest_commit_list, skipping.')
            continue

        url = MASTER_COMMIT_URL_TEMPLATE.replace('COMMIT_ID', commit)

        # Check if this version is newer than what we have
        if args.force or check_ov_version(exist_ov_version, url):
            ret_list.append(url)
        else:
            log.warning(f'Skipping older version. Have: {old_ov_version}, Found: {get_ov_version(url)} at {url}')

        # Stop once we have 10 valid, new URLs
        if len(ret_list) >= 10:
            break

    log.info('Found valid URL list from openvino/master:')
    if not ret_list:
        log.error(f'No new valid URLs found.')
    else:
        for url in ret_list:
            log.info(f'\t{url}')
    return ret_list, old_ov_version != None

def get_url(commit_id: str) -> str | None:
    """
    Searches known build server paths for a specific commit ID prefix.
    """
    url_list = [
        'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/commit',
        'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/pre_commit',
        'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2025/4/commit',
        'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2025/3/commit',
    ]

    for url in url_list:
        log.info(f'Querying commit list from {url}')
        root = get_text_from_web(url)
        if not root:
            continue

        soup = BeautifulSoup(root, 'html.parser')
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                match_obj = re.search(r'([\w]+)\/', href) # Find 'commit_hash/'
                if match_obj:
                    full_commit_id = match_obj.groups()[0]
                    if full_commit_id.startswith(commit_id):
                        ret_url = f'{url}/{full_commit_id}/{get_private_cpack_url()}'
                        log.info(f'Found URL: {ret_url}')
                        return ret_url
    log.error(f'Could not find any URL for commit prefix: {commit_id}')
    return None

def generate_manifest_url_from_cpack(cpack_url: str):
    index = cpack_url.rfind('cpack')
    if index > 0:
        return f'{cpack_url[:index]}/manifest.yml'
    else:
        return ''

def save_manifest(cpack_url: str, new_out_path: Path) -> Path | None:
    """Downloads the manifest.yml file from the build directory."""
    url = generate_manifest_url_from_cpack(cpack_url)
    return download_file(url, new_out_path)

def generate_manifest(manifest_path: Path) -> str:
    """Parses a manifest.yml file and returns a formatted markdown table."""
    if check_filepath(manifest_path):
        try:
            with open(manifest_path, 'rt', encoding='utf-8') as fis:
                raw_data_list = []
                data_dic = yaml.safe_load(fis.read())

                # Filter for specific repositories
                for repo in data_dic['components']['dldt']['repository']:
                    if repo["name"] in ['openvino', 'openvino_tokenizers', 'openvino.genai']:
                        raw_data_list.append([repo["name"], repo["url"], repo["branch"], repo["revision"]])

                headers = ['name', 'url', 'branch', 'revision']
                return tabulate(raw_data_list, tablefmt="github", headers=headers, stralign='left')
        except (IOError, yaml.YAMLError, KeyError) as e:
            log.error(f'Failed to parse manifest {manifest_path}: {e}')
    return ''


################################################
# Main
################################################
def main():
    log.basicConfig(level=log.INFO, format='[%(filename)s:%(lineno)4s:%(funcName)20s] %(levelname)s: %(message)s')

    # ... (Help text definition) ...
    help_download_url = "..." # (omitted for brevity)

    parser = argparse.ArgumentParser(description="download ov nightly" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Use Path object directly for output argument
    parser.add_argument('-o', '--output', help='openvino package stored directory', type=Path, default=PWD / 'openvino_nightly')
    parser.add_argument('--manifest', help='print manifest from a path', type=Path, default=None)
    parser.add_argument('-d', '--download_url', help=help_download_url, type=str, default=None)
    parser.add_argument('-c', '--commit_id', help='commit id for openvino', type=str, default=None)
    parser.add_argument('--keep_old', help='keep old pkg files/directories', action='store_true')
    parser.add_argument('--no_proxy', help='try to download pkgs with no_proxy', action='store_true')
    parser.add_argument('-i', '--install', help='install openvino package from file or dir', type=Path, default=None)
    parser.add_argument('--latest_commit', help='query latest master commit', action='store_true')
    parser.add_argument('--force', help='download ov latest pkg even if it already downloaded', action='store_true')
    args = parser.parse_args()

    # --- Mode 1: Manifest Generation ---
    if args.manifest:
        log.info('\n' + generate_manifest(args.manifest))
        sys.exit(0)

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # --- Mode 2: Local Install ---
    if args.install:
        if args.install.is_dir():
            # Install all archives in the specified directory
            for file_path in list(args.install.glob('*.zip')) + list(args.install.glob('*.tgz')):
                install_openvino(file_path, args.output)
        elif args.install.is_file():
            # Install a single specified archive
            install_openvino(args.install, args.output)
        else:
            log.error(f'Install path not found: {args.install}')
            sys.exit(-1)
        sys.exit(0)

    # --- Mode 3: Download & Install ---

    # [User Request] WA: replace 'https' to 'http'
    if args.download_url:
        args.download_url = args.download_url.replace('https', 'http')

    # WA: add no_proxy to environment
    if args.no_proxy:
        os.environ['no_proxy'] = 'localhost,intel.com,192.168.0.0/16,172.16.0.0/12,127.0.0.0/8,10,0.0.0/8'

    # Case 1: URL is a direct .zip or .tgz file
    if args.download_url and (args.download_url.endswith('.zip') or args.download_url.endswith('.tgz')):
        ov_zip_filepath = download_file(args.download_url, args.output)
        if ov_zip_filepath:
            install_openvino(ov_zip_filepath, args.output)
            sys.exit(0)
        else:
            log.error(f'Failed to download file: {args.download_url}')
            sys.exit(-1)

    # Case 2: URL is a cpack dir, commit ID, or find latest
    else:
        if args.download_url:
            target_url_list = [ args.download_url ]
        elif args.commit_id:
            url = get_url(args.commit_id)
            target_url_list = [ url ] if url else []
        else:
            target_url_list, exist_ov = get_list_of_openvino_master(args)
            if exist_ov and not target_url_list:
                # We have an old version, and no new versions were found
                log.info('No new versions found. Exiting.')
                sys.exit(0) # Not an error, just no new work

        if not target_url_list:
            log.error('No target URLs specified or found.')
            sys.exit(-1)

        # Try each found URL until one succeeds
        for target_url in target_url_list:
            log.info(f'--- Attempting to download from {target_url} ---')
            try:
                if not check_required_packages(target_url):
                    log.warning(f'-> Missing required packages at {target_url}, trying next URL.')
                    continue

                if args.latest_commit:
                    log.info(f'-> Found valid latest commit with all packages: {target_url}')
                    sys.exit(0) # Success for 'latest_commit' mode

                log.info(f'- Downloading OpenVINO packages...')
                openvino_zip_file_list, new_out_path = download_openvino_packages(target_url, args.output)

                log.info(f'- Downloading GenAI packages...')
                genai_zip_list = download_genai_packages(target_url, new_out_path)

                all_zips = openvino_zip_file_list + genai_zip_list
                if not all_zips:
                    raise Exception("No files were successfully downloaded.")

                log.info(f'- Decompressing {len(all_zips)} zip files...')
                # Use synchronous tqdm for the synchronous loop
                for zip_file in tqdm(all_zips):
                    decompress(zip_file, zip_file.parent, delete_zip=True)

                # Update pointers and save metadata
                setup_script = new_out_path / ('setupvars.bat' if IS_WINDOWS else 'setupvars.sh')
                update_latest_ov_setup_file(setup_script, args.output)
                save_ov_version(args, get_ov_version(target_url))

                manifest_filepath = save_manifest(target_url, new_out_path)
                if manifest_filepath:
                    log.info('--- Manifest ---')
                    log.info(generate_manifest(manifest_filepath))
                    log.info('----------------')

                log.info(f'Successfully downloaded and unzipped packages to: {new_out_path}')
                break  # Success, exit the loop

            except Exception as e:
                log.warning(f'Failed to process URL {target_url}: {e}')
                # If this was the last URL, exit with an error
                if target_url == target_url_list[-1]:
                    raise  # Re-raise the last exception

    # --- Cleanup ---
    if not args.keep_old:
        log.info(f'Cleaning up old files in {args.output}...')
        # No os.chdir() needed. Operate on args.output Path object

        # Remove old archives
        for file in list(args.output.glob('*.zip')) + list(args.output.glob('*.tgz')):
            log.info(f'Removing archive: {file.name}')
            try:
                os.remove(file)
            except OSError as e:
                log.warning(f'Could not remove {file.name}: {e}')

        # Remove old directories, keeping the N most recent
        REMAINING_DIR_COUNT = 20
        # Get all subdirectories, not files
        dir_list = [d for d in args.output.iterdir() if d.is_dir()]
        # Sort by modification time, newest first
        dir_list.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Delete directories *after* the Nth one
        if len(dir_list) > REMAINING_DIR_COUNT:
            for old_dir in dir_list[REMAINING_DIR_COUNT:]:
                log.info(f'Removing old directory: {old_dir.name}')
                try:
                    shutil.rmtree(old_dir)
                except OSError as e:
                    log.warning(f'Could not remove {old_dir.name}: {e}')

    sys.exit(0)  # Explicitly exit with success code


if __name__ == "__main__":
    main()
