# python preprocess/py ./raw ./cleaned

import pandas as pd
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('raw_folder_path', type=str, default="./raw/",
                    help='the path of raw csv folder.')
parser.add_argument('cleaned_folder_path', type=str, default="./cleaned/",
                    help='the path of cleaned csv folder.')
args = parser.parse_args()

files = glob.glob(f'{args.raw_folder_path}/*.csv')

def convert_count_to_int(value):
    value = value.lower()
    try:
        value = int(value)
    except:
        if 'm' in value:
            value = int(float(value.replace('m', '')) * 1e6)
        elif 'k' in value:
            value = int(float(value.replace('k', '')) * 1e3)
        else:
            value = -1
    return value

def get_video_time(value):
    time_str = value.split('/')[1]
    min, sec = tuple(time_str.split(':'))
    return int(min)*60 + int(sec)


for file in files:
    f_name = file.replace(f'{args.raw_folder_path}\\', '')
    df = pd.read_csv(file, encoding='utf-8')
    # convert like_count to int
    df['like_count'] = df['like_count'].apply(lambda v: convert_count_to_int(v))
    # convert comment_count to int
    df['comment_count'] = df['comment_count'].apply(lambda v: convert_count_to_int(v))
    # get video time
    df['video_time'] = df['time_container'].apply(lambda v: get_video_time(v))
    # calculate watched_time
    df['watched_time'] = -1 * df['timestamp'].diff(periods=-1)

    cleaned_df = df[['url','desc_video','like_count','comment_count','like','video_time','watched_time']]
    cleaned_df.to_csv(f'{args.cleaned_folder_path}/cleaned_{f_name}', index=False)



