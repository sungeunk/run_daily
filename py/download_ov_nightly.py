import argparse
import enum
import os
import platform
import re
import requests
import subprocess

from dateutil.parser import parse
from pathlib import Path
from tqdm import tqdm

################################################
# Global variable
################################################
IS_WINDOWS = platform.system() == 'Windows'
PWD = os.path.abspath('.')
OV_PRIVATE_MASTER_URL = 'http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/master/commit'

if not IS_WINDOWS:
    import lsb_release
    UBUNTU_VER = lsb_release.get_os_release()["RELEASE"].replace('.', '_')
else:
    UBUNTU_VER = ''



def get_text_from_web(url):
    res = requests.get(url)
    if res.ok:
        return res.text
    else:
        print(f'Could not get file: res({res.status_code}/{res.reason}) {url}')
    return None

def get_file_from_web(url, out_path):
    res = requests.get(url, stream=True)
    if res.ok:
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
        print(f'Could not get file: res({res.status_code}/{res.reason}) {url}')
    return None

def parse_commit_id_from_last_build_file(str):
    if str != None and len(str) > 0:
        match_obj = re.search(r'commit[\\:;/]([\da-z]+)', str)
        if match_obj != None:
            return match_obj.groups()[0]
    return None

def parse_commit_id_list_from_html(html):
    ret_list = []
    for line in html.splitlines():
        match_obj1 = re.search(r'href="([a-zA-Z\d]+)\/', line)
        match_obj2 = re.search(r'([\d]+-[a-zA-Z]+-[\d]+ [\d]+:[\d]+)', line)
        if match_obj1 != None and match_obj2 != None:
            commit_id = match_obj1.groups()[0]
            date = parse(match_obj2.groups()[0])
            ret_list.append((commit_id, date))

    ret_list = sorted(ret_list, key=lambda commit_list: commit_list[1], reverse=True)
    commit_list = []
    for commit, date in ret_list:
        commit_list.append(commit)

    return commit_list[:10]

def parse_package_filename(args, str):
    if str != None and len(str) > 0:
        for line in str.splitlines():
            match_obj = re.search(r'<a\s+(?:[^>]*?\s+)?href=(["\'])(.*?)\1', line)
            if match_obj != None:
                match_str = match_obj.groups()[1]
                if args.windows_target and match_str[-3:] == 'zip':
                    return match_str
                elif not args.windows_target and match_str.find(f'ubuntu{UBUNTU_VER[:2]}') != -1:
                    return match_str
    return None

def decompress(args, compressed_filepath, store_path):
    root, ext = os.path.splitext(compressed_filepath)
    if ext == '.zip':
        import zipfile
        with zipfile.ZipFile(compressed_filepath, 'r') as file:
            file.extractall(store_path)
    elif ext == '.tgz':
        import tarfile
        with tarfile.open(compressed_filepath) as file:
            file.extractall(store_path)

    return os.path.join(*[store_path, os.path.basename(root)]), ext

def download_openvino(args):
    download_filepath = None
    last_build_filename = 'last_build_private_windows_vs2019' if args.windows_target else f'last_build_private_linux_ubuntu_{UBUNTU_VER}'

    # generate try_commit_id_list from web.
    if args.commit_id == None:
        text = get_text_from_web(f'{OV_PRIVATE_MASTER_URL}/{last_build_filename}')
        commit_id = parse_commit_id_from_last_build_file(text)

        # parse commit_id from f'{OV_PRIVATE_MASTER_URL}'
        #    -> get latest 10 commits
        text = get_text_from_web(f'{OV_PRIVATE_MASTER_URL}')
        commit_id_list = parse_commit_id_list_from_html(text)

        try_commit_id_list = [commit_id]
        try_commit_id_list.extend(commit_id_list)
    else:
        try_commit_id_list = [args.commit_id]

    # try to download from try_commit_id_list
    for commit_id in try_commit_id_list:
        if commit_id == None: continue

        package_directory_url = f'{OV_PRIVATE_MASTER_URL}/{commit_id}/swf_drop/packages/releases/'
        text = get_text_from_web(package_directory_url)
        filename = parse_package_filename(args, text)
        if filename == None: continue

        download_filepath = get_file_from_web(package_directory_url + filename, args.openvino_nightly_dir)
        if download_filepath == None: continue

        print(f'Downloaded ov package: {download_filepath}')
        if args.download_only:
            break

        install_openvino(args, download_filepath)
        break

def install_openvino(args, ov_filepath=''):
    if ov_filepath == '' and args.install_ov_package != None:
        ov_filepath = args.install_ov_package

    uncompressed_dir, ext = decompress(args, ov_filepath, args.openvino_nightly_dir)
    setup_script = os.path.join(*[uncompressed_dir, 'setupvars.bat' if ext == '.zip' else 'setupvars.sh'])
    latest_ov_setup_file = os.path.join(*[args.openvino_nightly_dir, 'latest_ov_setup_file.txt'])

    with open(latest_ov_setup_file, 'w') as fos:
        fos.write(f'{setup_script}')
        print(f'Updated: {latest_ov_setup_file}')
        print(f'New setup script: {setup_script}')

    if not args.download_only and os.path.exists(latest_ov_setup_file):
        print(f'Please setup with exist ov package: {latest_ov_setup_file}')

################################################
# Main
################################################
def main():
    parser = argparse.ArgumentParser(description="download ov nightly" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ond', '--openvino_nightly_dir', help='openvino nightly package stored directory', type=Path, default=os.path.join(*[PWD, 'openvino_nightly']))
    parser.add_argument('-c', '--commit_id', help='target commit id for openvino nightly', type=str, default=None)
    parser.add_argument('-do', '--download_only', help='download ov package only. Do not decompress ov package', action='store_true')
    parser.add_argument('-wt', '--windows_target', help='download ov package for windows.', action='store_true', default=IS_WINDOWS)
    parser.add_argument('-i', '--install_ov_package', help='install ov package from compressed file.', type=Path, default=None)
    args = parser.parse_args()

    os.makedirs(args.openvino_nightly_dir, exist_ok=True)

    if args.install_ov_package:
        install_openvino(args)
    else:
        download_openvino(args)



if __name__ == "__main__":
    main()
