
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
    import tqdm.asyncio as tqdm_asyncio
    import yaml

    from bs4 import BeautifulSoup
    from dateutil.parser import parse
    from git import Repo, RemoteProgress
    from glob import glob
    from packaging.version import Version
    from pathlib import Path
    from tabulate import tabulate
    from tqdm import tqdm
except ImportError:
    print(f'Please install these modules: pip install aiohttp requests tqdm pyyaml bs4 python-dateutil gitpython packaging tabulate')
    sys.exit(-1)

from common_utils import *


################################################
# Global variable
################################################
IS_WINDOWS = platform.system() == 'Windows'
PWD = os.path.abspath('.')
STORAGE_OV_FILETREE_JSON = 'https://storage.openvinotoolkit.org/filetree.json'


if IS_WINDOWS:
    UBUNTU_VER = ''
else:
    output = subprocess.check_output(['lsb_release', '-r'], text=True)
    match_obj = re.search(r'Release:[ \t]+(\d+).(\d+)', output)
    if match_obj != None:
        UBUNTU_VER = f'{match_obj.groups()[0]}_{match_obj.groups()[1]}'

def check_filepath(path):
    try:
        return os.path.exists(path)
    except:
        log.error(f'could not find: {path}')
    return False

def convert_path(path):
    return path.replace('/', '\\') if IS_WINDOWS else path.replace('\\', '/')

def get_text_from_web(url):
    try:
        res = requests.get(url, verify=False)
        if res.ok:
            return res.text
        else:
            log.error(f'Could not get file: res({res.status_code}/{res.reason}) {url}')
            return ''
    except Exception as e:
        log.error(f'get_text_from_web::Exception: {e}')
        return ''

def download_file(url, out_path):
    res = requests.get(url, stream=True)
    if res.ok:
        os.makedirs(out_path, exist_ok=True)

        total = int(res.headers.get('content-length', 0))
        filename = os.path.basename(url)
        filepath = os.path.join(out_path, filename)

        with open(filepath, 'wb') as file, tqdm_asyncio.tqdm(
            desc=filename,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in res.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
            return os.path.realpath(filepath)
    else:
        log.error(f'Could not get file: res({res.status_code}/{res.reason}) {url}')
    return None

async def async_download_files(url_list, out_path):
    semaphore = asyncio.Semaphore(5)

    async def async_download_file(session, url, out_path):
        async with semaphore:
            async with session.get(url) as res:
                if res.ok:
                    filename = os.path.basename(url)
                    filepath = os.path.join(out_path, filename)

                    with open(filepath, 'wb') as file:
                        async for data in res.content.iter_chunked(1024*4):
                            file.write(data)
                        return os.path.realpath(filepath)
                else:
                    raise Exception(f'Could not get file: res({res.status_code}/{res.reason}) {url}')

    async with aiohttp.ClientSession() as session:
        return await tqdm_asyncio.tqdm.gather(
            *[async_download_file(session, url, out_path) for url in url_list]
        )

def required_openvino_packages_list():
    REQUIRED_LIST = [
        'benchmark_app.zip',
        'core.zip',
        'core_c.zip',
        'core_c_dev.zip',
        'core_dev.zip',
        'cpp_samples.zip',
        'cpu.zip',
        'gpu.zip',
        'ir.zip',
        'onnx.zip',
        'openvino_req_files.zip',
        'ovc.zip',
        'paddle.zip',
        'pyopenvino_python3.10.zip',
        'pyopenvino_python3.11.zip',
        'pyopenvino_python3.12.zip',
        'pytorch.zip',
        'setupvars.zip',
        'tbb.zip',
        'tbb_dev.zip',
        'tensorflow.zip',
        'tensorflow_lite.zip',
        ]
    return REQUIRED_LIST

def required_genai_packages_list():
    REQUIRED_LIST = [
        'openvino_tokenizers.zip',
        'core_genai.zip',
        'core_genai_dev.zip',
        'pygenai_3_10.zip',
        'pygenai_3_11.zip',
        'pygenai_3_12.zip',
        ]
    return REQUIRED_LIST

def check_required_packages(url):
    check_list = required_openvino_packages_list() + required_genai_packages_list()

    text = get_text_from_web(url)
    if len(text) == 0:
        return False

    for item in check_list:
        match_obj = re.search(item, text)
        if match_obj == None:
            return False

    return True

def get_ov_version(url):
    text = get_text_from_web(url)
    match_obj = re.search(r'inference-engine_Release-(\d+.\d+.\d+.\d+)-', text)
    return match_obj.groups()[0] if match_obj else ''

def save_ov_version(args, new_version):
    OLD_VERSION_FILEPATH = convert_path(f'{args.output}/OV_NIGHTLY_VERSION')
    with open(convert_path(OLD_VERSION_FILEPATH), 'w', encoding='utf8') as fos:
        fos.write(new_version)

def load_ov_version(args) -> str:
    try:
        OLD_VERSION_FILEPATH = convert_path(f'{args.output}/OV_NIGHTLY_VERSION')
        with open(convert_path(OLD_VERSION_FILEPATH), 'r', encoding='utf8') as fis:
            return fis.readline()
    except Exception as e:
        return ''

def check_ov_version(args, url):
    try:
        old_version = Version(load_ov_version(args))
    except Exception as e:
        log.error(f'check_ov_version::try to load old version: {e}')
        return True

    try:
        version_str = get_ov_version(url)
        if len(version_str) == 0:
            return False
        new_version = Version(version_str)
    except Exception as e:
        log.error(f'check_ov_version::try to get new version from: {url} / log: {e}')
        log.error(f'\t{e}')
        return False

    return new_version > old_version


# Target urls
# http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/custom_build/6c58834d4b4e79605ada3ae9ee04b4a56cabd227/private_windows_vs2019_release/cpack/
# http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/pre_commit/05f042e882ed2627e7b6cfc7521244dfc879a506/private_windows_vs2019_release/cpack/
def download_openvino_packages(url, out_path):
    REQUIRED_LIST = required_openvino_packages_list()

    COMMIT_ID = None
    match_obj = re.search(r'(custom_build|commit|pre_commit)\/([\w]+)', url)
    if match_obj:
        COMMIT_ID = match_obj.groups()[1]
    if not COMMIT_ID:
        raise Exception(f'could not parse COMMIT_ID url: {url}')

    text = get_text_from_web(url)
    log.debug(f'root: {text}')

    if IS_WINDOWS:
        #                        inference-engine_Release-2025.1.0.18201-win64-benchmark_app.zip
        match_obj = re.search(fr'inference-engine_Release-(\d+.\d+.\d+.\d+)-win64-{REQUIRED_LIST[0]}', text)
    else:
        #                        inference-engine-2025.1.0.18206-Linux-benchmark_app.zip
        match_obj = re.search(fr'inference-engine-(\d+.\d+.\d+.\d+)-Linux-{REQUIRED_LIST[0]}', text)
    if not match_obj:
        raise Exception(f'could not parse the OV_VERSION from text: {text}')

    OV_VERSION = match_obj.groups()[0]
    new_out_path = os.path.join(*[out_path, f'{OV_VERSION}_{COMMIT_ID[0:8]}'])
    os.makedirs(new_out_path, exist_ok=True)
    if IS_WINDOWS:
        list = [ f'{url}/inference-engine_Release-{OV_VERSION}-win64-{target_filename}' for target_filename in REQUIRED_LIST ]
    else:
        list = [ f'{url}/inference-engine-{OV_VERSION}-Linux-{target_filename}' for target_filename in REQUIRED_LIST ]
    return asyncio.run(async_download_files(list, new_out_path)), new_out_path

def download_genai_packages(url, ov_dst_path):
    REQUIRED_LIST = required_genai_packages_list()

    text = get_text_from_web(url)
    os_name = 'win64' if IS_WINDOWS else 'Linux'
    match_obj = re.search(fr'OpenVINOGenAI-([\d.]+)-{os_name}-{REQUIRED_LIST[0]}', text)
    if not match_obj:
        raise Exception(f'could not parse the GENAI_VERSION from text: {text}')

    GENAI_VERSION = match_obj.groups()[0]
    list = [ f'{url}/OpenVINOGenAI-{GENAI_VERSION}-{os_name}-{target_filename}' for target_filename in REQUIRED_LIST ]
    return asyncio.run(async_download_files(list, ov_dst_path))

def decompress(compressed_filepath, store_path, delete_zip=False):
    old_path = os.curdir
    os.chdir(store_path)

    root, ext = os.path.splitext(compressed_filepath)
    if ext == '.zip':
        if IS_WINDOWS:
            import zipfile
            with zipfile.ZipFile(compressed_filepath, 'r') as file:
                file.extractall(store_path)
        else:
            os.system(f'unzip -o -q {compressed_filepath}')
    elif ext == '.tgz':
        import tarfile
        with tarfile.open(compressed_filepath) as file:
            file.extractall(store_path)

    if delete_zip:
        os.remove(compressed_filepath)

    os.chdir(old_path)

    return os.path.join(*[store_path, os.path.basename(root)]), ext

def install_openvino(ov_filepath, output):
    if not check_filepath(ov_filepath):
        log.warning(f'no file: {ov_filepath}')
        return

    uncompressed_dir, ext = decompress(ov_filepath, output)
    setup_script = os.path.join(*[uncompressed_dir, 'setupvars.bat' if IS_WINDOWS else 'setupvars.sh'])
    update_latest_ov_setup_file(setup_script, output)

def update_latest_ov_setup_file(setup_script_path, output):
    latest_ov_setup_file = os.path.join(*[output, 'latest_ov_setup_file.txt'])

    with open(latest_ov_setup_file, 'w') as fos:
        fos.write(f'{setup_script_path}')
        log.info(f'New setup script: {setup_script_path}')
        log.info(f'Please setup with exist ov package: {latest_ov_setup_file}')

def get_child(json_obj, name):
    for child_obj in json_obj["children"]:
        if child_obj != None and child_obj["name"] == name:
            return child_obj
    return None

def get_directory(json_obj, directory):
    dirs = directory.split('/')
    for dir_name in dirs:
        json_obj = get_child(json_obj, dir_name.strip())
        if json_obj == None:
            return None

    return json_obj

def get_private_cpack_url():
    if IS_WINDOWS:
        return 'private_windows_vs2022_release/cpack'
    else:
        return f'private_linux_ubuntu_{UBUNTU_VER}_release/cpack'

def get_latest_commit_list_from_openvino():
    OPENVINO_URL = 'https://github.com/openvinotoolkit/openvino.git'
    OPENVINO_DIR = 'openvino.git'
    try:
        repo = Repo(OPENVINO_DIR)
    except Exception as e:
        log.error(f'{e.__class__}: {e}')
        log.debug(f'try to clone --base --single-branch {OPENVINO_URL}')
        repo = Repo.clone_from(url=OPENVINO_URL, to_path=OPENVINO_DIR, progress=CloneProgress(), multi_options=["--bare", "--single-branch"])

    repo.remote().fetch("refs/heads/master:refs/heads/origin")
    git = repo.git
    git.reset('FETCH_HEAD', '--soft')

    commit_list = []
    for commit in repo.iter_commits():
        # commit_list.append((str(commit), commit.count()))
        commit_list.append(str(commit))
        if len(commit_list) >= 10:
            break
    repo.close()
    return commit_list

#
# ret_items: [(build_number, commit-id, json_obj), ...]
#
def get_list_of_openvino_master(args):
    MASTER_COMMIT_URL_ROOT = 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/commit'
    log.info(f'query openvino list: {MASTER_COMMIT_URL_ROOT}')

    root = get_text_from_web(MASTER_COMMIT_URL_ROOT)
    MASTER_COMMIT_URL_TEMPLATE=f'{MASTER_COMMIT_URL_ROOT}/COMMIT_ID/{get_private_cpack_url()}'
    master_commit_list = []
    for line in root.splitlines():
        # <a href="fa6b0ec1715d9f09ef17a9ac1a052e4c20781288/">fa6b0ec1715d9f09ef17a9ac1a052e4c20781288/</a>          15-Nov-2024 08:30
        # <a href="faa6c75a225a775d6b037300d6947c0fb2cba7f2/">faa6c75a225a775d6b037300d6947c0fb2cba7f2/</a>          14-Nov-2024 05:57
        match_obj = re.search(r'<a href="([0-9a-z]+)\/">[0-9a-z]+\/<\/a> +([\d]+-[a-zA-Z]+-[\d]+ [\d]+:[\d]+)', line)
        if match_obj:
            values = match_obj.groups()
            if len(values[0]) < 20: continue

            dateobj = datetime.datetime.strptime(values[1], '%d-%b-%Y %H:%M')
            master_commit_list.append([values[0], dateobj])
    master_commit_list.sort(key = lambda x : x[1], reverse=True)

    old_ov_version = load_ov_version(args)
    exist_ov = len(old_ov_version) > 0
    latest_commit_list = get_latest_commit_list_from_openvino()
    ret_list = []
    for commit in master_commit_list[:40]:
        if not commit[0] in latest_commit_list:
            log.debug(f'{commit[0]} is not in latest_commit_list')
            continue
        url = MASTER_COMMIT_URL_TEMPLATE.replace('COMMIT_ID', commit[0])
        if check_ov_version(args, url):
            ret_list.append(url)
        else:
            log.warning(f'Exist ov version: {old_ov_version}. version is ({get_ov_version(url)}) in {url}')
        if len(ret_list) >= 10:
            break

    log.info('url list from openvino/master')
    if len(ret_list) == 0:
        log.error(f'no url list')
    else:
        for url in ret_list:
            log.info(f'\t{url}')
    return ret_list, exist_ov

def get_url(commit_id):
    url_list = {
        'master/commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/commit/',
        'master/pre_commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/pre_commit/',
        'releases/2025/2/commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2025/2/commit/',
        'releases/2025/1/commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2025/1/commit/',
        'releases/2025/0/commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2025/0/commit/',
        # 'releases/2024/4/commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2024/4/commit/',
        # 'releases/2024/4/pre_commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2024/4/pre_commit/',
        # 'releases/2024/5/commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2024/5/commit/',
        # 'releases/2024/5/pre_commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2024/5/pre_commit/',
        # 'releases/2024/6/commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2024/6/commit/',
        # 'releases/2024/6/pre_commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2024/6/pre_commit/',
        # 'releases/2025/0/pre_commit': 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2025/0/pre_commit/'
    }

    for key, url in url_list.items():
        log.info(f'query commit list from {url}')
        root = get_text_from_web(url)
        soup = BeautifulSoup(root, 'html.parser')
        # <a href="fa6b0ec1715d9f09ef17a9ac1a052e4c20781288/">fa6b0ec1715d9f09ef17a9ac1a052e4c20781288/</a>          15-Nov-2024 08:30
        # <a href="faa6c75a225a775d6b037300d6947c0fb2cba7f2/">faa6c75a225a775d6b037300d6947c0fb2cba7f2/</a>          14-Nov-2024 05:57
        for line in soup.find_all('a'):
            match_obj = re.search(r'([\w]+)\/', line["href"])
            if match_obj:
                if match_obj.groups()[0].startswith(commit_id):
                    ret_url = f'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/{key}/{match_obj.groups()[0]}/{get_private_cpack_url()}'
                    log.info(f'found url: {ret_url}')
                    return ret_url

def save_manifest(cpack_url, new_out_path):
    index = cpack_url.rfind('cpack')
    if index > 0:
        return download_file(f'{cpack_url[:index]}/manifest.yml', new_out_path)
    return ''

def generate_manifest(manifest_path) -> str:
    if check_filepath(manifest_path):
        with open(manifest_path, 'rt') as fis:
            raw_data_list = []
            data_dic = yaml.safe_load(fis.read())
            for repo in data_dic['components']['dldt']['repository']:
                if repo["name"] in ['openvino', 'openvino_tokenizers', 'openvino.genai']:
                    raw_data_list.append([repo["name"], repo["url"], repo["branch"], repo["revision"]])

            headers = ['name', 'url', 'branch', 'revision']
            return tabulate(raw_data_list, tablefmt="github", headers=headers, stralign='left')
    return ''

class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()

################################################
# Main
################################################
def main():
    log.basicConfig(level=log.INFO, format='[%(filename)s:%(lineno)4s:%(funcName)20s] %(levelname)s: %(message)s')

    help_download_url = """
    1. Download packages from cpack(OV/genai/tokenizer): http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/commit/ef5678a1098da18c3324a26392236d7974ed1cf5/private_windows_vs2019_release/cpack/\n
    2. Download zip file(OV only) : https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2025.0.0-17738-638f3cb9292/w_openvino_toolkit_windows_2025.0.0.dev20250102_x86_64.zip
    """

    parser = argparse.ArgumentParser(description="download ov nightly" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='openvino package stored directory', type=Path, default=os.path.join(*[PWD, 'openvino_nightly']))
    parser.add_argument('--manifest', help='print manifest', type=Path, default=None)
    parser.add_argument('-d', '--download_url', help=help_download_url, type=str, default=None)
    parser.add_argument('-c', '--commit_id', help='commit id for openvino', type=str, default=None)
    parser.add_argument('--keep_old', help='keep old pkg files/directories', action='store_true')
    parser.add_argument('--clean_up', help='[deprecated] not working. remove old pkg files/directories', type=bool, default=True)
    parser.add_argument('--no_proxy', help='try to download pkgs with no_proxy', action='store_true')
    parser.add_argument('-i', '--install', help='install openvino package', type=Path, default=None)
    parser.add_argument('--latest_commit', help='query latest master commit', action='store_true')
    args = parser.parse_args()

    if args.manifest != None:
        log.info('\n' + generate_manifest(args.manifest))
        return 0

    os.makedirs(args.output, exist_ok=True)

    if args.install:
        if os.path.isdir(args.install):
            os.chdir(args.install)
            for file in glob('*.zip') + glob('*.tgz'):
                decompress(file, args.output)
        else:
            install_openvino(args.install, args.output)
        return 0

    # WA: replace 'https' to 'http'
    if args.download_url:
        args.download_url = args.download_url.replace('https', 'http')

    # WA: add no_proxy to environment
    if args.no_proxy:
        os.environ['no_proxy'] = 'localhost,intel.com,192.168.0.0/16,172.16.0.0/12,127.0.0.0/8,10,0.0.0/8'

    retcode = -1
    try:
        if args.download_url and (args.download_url.endswith('.zip') or args.download_url.endswith('.tgz')):
            ov_zip_filepath = download_file(args.download_url, args.output)
            install_openvino(ov_zip_filepath, args.output)
            retcode = 0
        else:
            if args.download_url:
                target_url_list = [ args.download_url ]
            elif args.commit_id:
                target_url_list = [ get_url(args.commit_id) ]
            else:
                target_url_list, exist_ov = get_list_of_openvino_master(args)
                if exist_ov:
                    retcode = 0

            for target_url in target_url_list:
                log.info(f'try to download pkgs from {target_url}')
                try:
                    if not check_required_packages(target_url):
                        log.info(f'-> no required package in {target_url}')
                        continue

                    if args.latest_commit:
                        log.info(f'->    required package in {target_url}')
                        return 0

                    log.info(f'- download OpenVINO packages')
                    openvino_zip_file_list, new_out_path = download_openvino_packages(target_url, args.output)
                    log.info(f'- download GenAI packages')
                    genai_zip_list_list = download_genai_packages(target_url, new_out_path)

                    log.info(f'- decompress zip files')
                    for zip_file in tqdm_asyncio.tqdm(openvino_zip_file_list + genai_zip_list_list):
                        decompress(zip_file, os.path.dirname(zip_file), True)
                    update_latest_ov_setup_file(os.path.join(*[new_out_path, 'setupvars.bat' if IS_WINDOWS else 'setupvars.sh']), args.output)
                    save_ov_version(args, get_ov_version(target_url))
                    manifest_filepath = save_manifest(target_url, new_out_path)
                    log.info(f'{generate_manifest(manifest_filepath)}')
                    retcode = 0
                    break
                except Exception as e:
                    log.warning(f'{e}')
    except Exception as e:
        log.error(f'{e}')
        sys.exit(-1)

    # clean up old pkgs(zip/directory)
    if not args.keep_old:
        os.chdir(args.output)

        for file in glob('*.zip') + glob('*.tgz'):
            log.info(f'removed: {file}')
            os.remove(file)

        # remove all files/directories at output directory
        REMAINING_DIR_COUNT=20
        dir_list = os.listdir()
        dir_list.sort(key=os.path.getmtime, reverse=True)
        del dir_list[:REMAINING_DIR_COUNT]   # exclude 10 directories.
        for file in dir_list:
            log.info(f'removed: {file}')
            shutil.rmtree(file, ignore_errors=True)

    sys.exit(retcode)


if __name__ == "__main__":
    main()
