import aiohttp
import argparse
import asyncio
import datetime
import enum
import json
import logging as log
import os
import platform
import re
import requests
import shutil
import subprocess

from bs4 import BeautifulSoup
from dateutil.parser import parse
from glob import glob
from operator import itemgetter
from pathlib import Path
# from tqdm import tqdm
from tqdm.asyncio import tqdm

################################################
# Global variable
################################################
IS_WINDOWS = platform.system() == 'Windows'
PWD = os.path.abspath('.')
STORAGE_OV_FILETREE_JSON = 'https://storage.openvinotoolkit.org/filetree.json'

if not IS_WINDOWS:
    output = subprocess.check_output(['lsb_release', '-r'], text=True)
    match_obj = re.search(r'Release:[ \t]+(\d+.\d+)', output)
    if match_obj != None:
        UBUNTU_VER = match_obj.groups()[0].strip()
else:
    UBUNTU_VER = ''


def get_text_from_web(url):
    try:
        res = requests.get(url, verify=False)
        if res.ok:
            return res.text
        else:
            log.error(f'Could not get file: res({res.status_code}/{res.reason}) {url}')
            return None
    except Exception as e:
        log.error(f'get_text_from_web::Exception: {e}')
        return None

def download_file(url, out_path):
    res = requests.get(url, stream=True)
    if res.ok:
        os.makedirs(out_path, exist_ok=True)

        total = int(res.headers.get('content-length', 0))
        filename = os.path.basename(url)
        filepath = os.path.join(out_path, filename)

        with open(filepath, 'wb') as file, tqdm(
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
        return await tqdm.gather(
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
    for item in check_list:
        match_obj = re.search(item, text)
        if match_obj == None:
            return False

    return True


# Target urls
# http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/custom_build/6c58834d4b4e79605ada3ae9ee04b4a56cabd227/private_windows_vs2019_release/cpack/
# http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/pre_commit/05f042e882ed2627e7b6cfc7521244dfc879a506/private_windows_vs2019_release/cpack/
def download_openvino_packages(url, out_path):
    REQUIRED_LIST = required_openvino_packages_list()

    COMMIT_ID = None
    for name in ['commit', 'pre_commit', 'custom_build', 'nightly']:
        match_obj = re.search(f'{name}\/([\w\d]+)', url)
        if match_obj:
            COMMIT_ID = match_obj.groups()[0]
            break
    match_obj = re.search(f'releases\/([\d]+)\/([\d]+)\/[\w]+\/([\w]+)', url)
    if match_obj:
        COMMIT_ID = match_obj.groups()[2]
    if not COMMIT_ID:
        raise Exception(f'could not parse the commit id from url ({url})')

    text = get_text_from_web(url)
    log.debug(f'root: {text}')

    match_obj = re.search(f'inference-engine_Release-(\d+.\d+.\d+.\d+)-win64-{REQUIRED_LIST[0]}', text)
    if match_obj == None:
        raise Exception(f'could not parse the OV_VERSION from text: {text}')

    OV_VERSION = match_obj.groups()[0]
    new_out_path = os.path.join(*[out_path, f'{OV_VERSION}_{COMMIT_ID[0:8]}'])
    os.makedirs(new_out_path, exist_ok=True)
    list = [ f'{url}/inference-engine_Release-{OV_VERSION}-win64-{target_filename}' for target_filename in REQUIRED_LIST ]
    return asyncio.run(async_download_files(list, new_out_path)), new_out_path

def download_genai_packages(url, ov_dst_path):
    REQUIRED_LIST = required_genai_packages_list()

    text = get_text_from_web(url)
    match_obj = re.search(f'OpenVINOGenAI-([\d.]+)-win64-{REQUIRED_LIST[0]}', text)
    if match_obj == None:
        raise Exception(f'could not parse the GENAI_VERSION from text: {text}')

    GENAI_VERSION = match_obj.groups()[0]
    list = [ f'{url}/OpenVINOGenAI-{GENAI_VERSION}-win64-{target_filename}' for target_filename in REQUIRED_LIST ]
    return asyncio.run(async_download_files(list, ov_dst_path))

def decompress(compressed_filepath, store_path, delete_zip=False):
    root, ext = os.path.splitext(compressed_filepath)
    if ext == '.zip':
        import zipfile
        with zipfile.ZipFile(compressed_filepath, 'r') as file:
            file.extractall(store_path)
    elif ext == '.tgz':
        import tarfile
        with tarfile.open(compressed_filepath) as file:
            file.extractall(store_path)

    if delete_zip:
        os.remove(compressed_filepath)

    return os.path.join(*[store_path, os.path.basename(root)]), ext

def install_openvino(ov_filepath, output):
    if ov_filepath == None or not os.path.exists(ov_filepath):
        return

    uncompressed_dir, ext = decompress(ov_filepath, output)
    setup_script = os.path.join(*[uncompressed_dir, 'setupvars.bat' if ext == '.zip' else 'setupvars.sh'])
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

class TargetOS(enum.Enum):
    AUTO = 'auto'
    WINDOWS = 'windows'
    UBUNTU_20 = 'ubuntu20'
    UBUNTU_22 = 'ubuntu22'

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return TargetOS[s]
        except KeyError:
            raise ValueError()

#
# ret_items: [(build_number, commit-id, json_obj), ...]
#
def get_list_of_openvino_nightly():
    MASTER_COMMIT_URL_ROOT = 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/commit'
    log.info(f'query openvino list: {MASTER_COMMIT_URL_ROOT}')

    root = get_text_from_web(MASTER_COMMIT_URL_ROOT)
    soup = BeautifulSoup(root, 'html.parser')
    # <a href="fa6b0ec1715d9f09ef17a9ac1a052e4c20781288/">fa6b0ec1715d9f09ef17a9ac1a052e4c20781288/</a>          15-Nov-2024 08:30
    # <a href="faa6c75a225a775d6b037300d6947c0fb2cba7f2/">faa6c75a225a775d6b037300d6947c0fb2cba7f2/</a>          14-Nov-2024 05:57
    MASTER_COMMIT_URL_TEMPLATE=f'{MASTER_COMMIT_URL_ROOT}/COMMIT_ID/private_windows_vs2019_release/cpack'
    master_commit_list = []
    for line in soup.find_all('a'):
        match_obj = re.search(r'([\w]+)\/', line["href"])
        if match_obj:
            master_commit_list.append(match_obj.groups()[0])

    log.info(f'query openvino nightly list: {STORAGE_OV_FILETREE_JSON}')
    root = get_text_from_web(STORAGE_OV_FILETREE_JSON)
    root_json = json.loads(root)
    nightly_dir = get_directory(root_json, "repositories / openvino / packages / nightly")
    NIGHTLY_URL = 'https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly'
    nightly_url_list = []

    if UBUNTU_VER == '':
        target_os = TargetOS.WINDOWS
    elif UBUNTU_VER == '22.04':
        target_os = TargetOS.UBUNTU_22
    elif UBUNTU_VER == '22.04':
        target_os = TargetOS.UBUNTU_22

    for commit_dir in nightly_dir["children"]:
        if commit_dir["type"] == "directory":
            for file_json_obj in commit_dir["children"]:
                if file_json_obj["name"].endswith(('.tgz', '.zip')) and target_os.value in file_json_obj["name"]:
                    nightly_url_list.append((f'{NIGHTLY_URL}/{commit_dir["name"]}/{file_json_obj["name"]}', commit_dir["name"]))

    ret_list = []
    for url, version in sorted(nightly_url_list, reverse=True)[:10]:
        commit_id = ''
        match_obj = re.search(r'[\d.]+-[\d]+-([\w]+)', version)
        if match_obj:
            commit_id = match_obj.groups()[0]
            for master_commit in master_commit_list:
                if commit_id in master_commit:
                    ret_list.append(MASTER_COMMIT_URL_TEMPLATE.replace('COMMIT_ID', master_commit))

    return ret_list

#
# ret_items: [(build_number, commit-id, json_obj), ...]
#
def get_list_of_openvino_master():
    MASTER_COMMIT_URL_ROOT = 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/commit'
    log.info(f'query openvino list: {MASTER_COMMIT_URL_ROOT}')

    root = get_text_from_web(MASTER_COMMIT_URL_ROOT)
    MASTER_COMMIT_URL_TEMPLATE=f'{MASTER_COMMIT_URL_ROOT}/COMMIT_ID/private_windows_vs2019_release/cpack'
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

    ret_list = []
    for commit_id in master_commit_list[:10]:
        ret_list.append(MASTER_COMMIT_URL_TEMPLATE.replace('COMMIT_ID', commit_id[0]))
    return ret_list


################################################
# Main
################################################
def main():
    log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="download ov nightly" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='openvino package stored directory', type=Path, default=os.path.join(*[PWD, 'openvino_nightly']))
    parser.add_argument('-d', '--download_url', help='download file', type=str, default=None)
    parser.add_argument('-to', '--target_os', help='download ov package for OS. It\'s working on nightly links.', type=TargetOS.from_string, default=TargetOS.AUTO, choices=list(TargetOS))
    parser.add_argument('-i', '--install_zip', help='install ov package from compressed file.', type=Path, default=None)
    parser.add_argument('--clean_up', help='remove old pkg files/directories', type=bool, default=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ov_zip_filepath = None

    # WA: replace 'https' to 'http'
    if args.download_url != None:
        args.download_url = args.download_url.replace('https', 'http')

    try:
        if args.install_zip:
            ov_zip_filepath = args.install_zip
        else:
            if args.download_url and (args.download_url.endswith('.zip') or args.download_url.endswith('.tgz')):
                ov_zip_filepath = download_file(args.download_url, args.output)
            else:
                target_url_list = [ args.download_url ] if args.download_url else get_list_of_openvino_master()
                for target_url in target_url_list:
                    log.info(f'download pkgs from {target_url}')
                    try:
                        if not check_required_packages(target_url):
                            continue
                        log.info(f'- download OpenVINO packages')
                        openvino_zip_file_list, new_out_path = download_openvino_packages(target_url, args.output)
                        log.info(f'- download GenAI packages')
                        genai_zip_list_list = download_genai_packages(target_url, new_out_path)

                        log.info(f'- decompress zip files')
                        for zip_file in tqdm(openvino_zip_file_list + genai_zip_list_list):
                            decompress(zip_file, os.path.dirname(zip_file), True)
                        update_latest_ov_setup_file(os.path.join(*[new_out_path, 'setupvars.bat' if IS_WINDOWS else 'setupvars.sh']), args.output)
                        break
                    except Exception as e:
                        log.warning(f'{e}')
    except Exception as e:
        log.error(f'{e}')
        return

    # decompress zip file and update latest_ov_setup_file.txt
    if ov_zip_filepath and os.path.exists(ov_zip_filepath):
        install_openvino(ov_zip_filepath, args.output)

    # clean up old pkgs(zip/directory)
    if args.clean_up:
        os.chdir(args.output)

        for file in glob('*.zip'):
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



if __name__ == "__main__":
    main()