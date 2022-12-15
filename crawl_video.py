# python crawl_video.py -u_p './raw' -dest './videos' -vid 1 -thumb 1
import requests
import json
from urllib.request import urlretrieve
import os
import argparse
import pandas as pd
import numpy as np


def read_file(file_name):
    df = pd.read_csv(file_name)
    url = df['url'].to_list()
    return url

def load_url_file(url_path):
    if os.path.isfile(url_path):
        return read_file(url_path)

    if os.path.isdir(url_path):
        results = []
        dirnames = os.listdir(url_path)
        for dir in dirnames:
            path = os.path.join(url_path, dir)
            results += read_file(path)
        return results


def create_direction(path):
    is_exist_path = os.path.exists(path)
    if not is_exist_path:
        os.mkdir(path)


def request_url(url):
    req_url = "https://tik-tok-video.com/api/convert"
    req_payload = {
        "url": url
    }

    req_headers = {
        "content-type": "application/json",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }

    response = requests.request("POST", req_url, json=req_payload, headers=req_headers)
    return response


def download_video(response, dest_path, down_vid=False, down_thumb=False):
    print("***Request video info from url***")
    vid_info = json.loads(response.text)
    print(f"Video id: {vid_info['id']}")
    vid_mp4 = vid_info['url'][0]
    result = [vid_info['id'], vid_mp4['url'], vid_info['thumb']]

    if down_vid or down_thumb:
        vid_path = os.path.join(dest_path, vid_info['id'])
        create_direction(vid_path)
        if down_vid:
            vid_fname = f"{vid_path}/{vid_info['id']}.{vid_mp4['ext']}"
            if not os.path.exists(vid_fname):
                print('- Downloading video...')
                vid_resp_download = urlretrieve(vid_mp4['url'], vid_fname)
                print("\t=> Finish downloading video!")
        if down_thumb:
            thumb_fname = f"{vid_path}/{vid_info['id']}.png"
            if not os.path.exists(thumb_fname):
                print('- Downloading thumbnail image...')
                img_resp_download = urlretrieve(vid_info['thumb'], thumb_fname)
                print("\t=> Finish downloading thumbnail image!")
    print('-' * 50)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u_p', '--url_fname_path', type=str, default="./videos/list_url.txt",
                        help='the path of url.')
    parser.add_argument('-dest', '--destination', type=str, default="./videos",
                        help='the path of downloaded file.')
    parser.add_argument('-vid', '--download_video', type=bool, default=False,
                        help='download video?')
    parser.add_argument('-thumb', '--download_thumb', type=bool, default=False,
                        help='download thumbnail images?')
    args = parser.parse_args()

    urls = load_url_file(args.url_fname_path)
    download_urls = []
    for url in urls:
        respone = request_url(url)
        down_url = download_video(respone, args.destination, args.download_video, args.download_thumb)
        download_urls.append(down_url)
    df = pd.DataFrame(download_urls, columns=['id', 'url_video', 'url_thumb'])
    csv_path = os.path.join(args.destination, 'download_url.csv')
    df.to_csv(csv_path, index=False)
