# python preprocess.py -r ./raw -c ./cleaned -i ./input

import pandas as pd
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--raw', type=str, default="./raw/",
                    help='the path of raw csv folder.')
parser.add_argument('-c', '--cleaned', type=str, default="./cleaned/",
                    help='the path of cleaned csv folder.')
parser.add_argument('-i', '--input', type=str, default="./cleaned/",
                    help='the path of input-format csv folder.')
parser.add_argument('-s', '--for_server', action='store_true',
                    help='create data for server model.')
args = parser.parse_args()

raw_files = glob.glob(f'{args.raw}/*.csv')


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
    return int(min) * 60 + int(sec)

num_ord_candidates = 4
num_recommend = 5
num_candidates = 7

def do(num_watched_videos, for_server=False):
    # Tidy raw files
    for file in raw_files:
        f_name = file.replace(f'{args.raw}\\', '')
        df = pd.read_csv(file, encoding='utf-8')
        slide_time = (df.shape[0] >= (num_watched_videos+num_candidates))*(1 + ((df.shape[0]-(num_watched_videos+num_candidates)) // 5))
        if slide_time < 1:
            print(f'The number of watched videos in {f_name} is less than {num_watched_videos+num_candidates}!!!')
            continue
        # convert like_count to int
        df['like_count'] = df['like_count'].astype('str')
        df['like_count'] = df['like_count'].apply(lambda v: convert_count_to_int(v))
        # convert comment_count to int
        df['comment_count'] = df['comment_count'].astype('str')
        df['comment_count'] = df['comment_count'].apply(lambda v: convert_count_to_int(v))
        # get video time
        df['video_time'] = df['time_container'].apply(lambda v: get_video_time(v))
        # calculate watched_time
        df['watched_time'] = -1 * df['timestamp'].diff(periods=-1)

        cleaned_df = df[['url', 'desc_video', 'like_count', 'comment_count', 'like', 'video_time', 'watched_time', 'user']]
        cleaned_df.to_csv(f'{args.cleaned}/cleaned_{f_name}', index=False)

    # # Merge cleaned files
    # cleaned_files = glob.glob(f'{args.cleaned}/*.csv')
    # merged_df = pd.DataFrame(columns=[
    #     'url','desc_video','like_count','comment_count','like','video_time','watched_time','user'
    # ])

    # for file in cleaned_files:
    #     cleaned_df = pd.read_csv(file, encoding='utf-8')
    #     merged_df = pd.concat([merged_df, cleaned_df])

    # merged_df.to_csv(f'{args.cleaned}/final.csv', index=False)

    # Create input files from cleaned files
    cleaned_files = glob.glob(f'{args.cleaned}/*.csv')
    input_df = pd.DataFrame(columns=[
        'uid', \
        'w1_duration', 'w1_num_likes', 'w1_num_comments', 'w1_watched_time', 'w1_like', \
        'w2_duration', 'w2_num_likes', 'w2_num_comments', 'w2_watched_time', 'w2_like', \
        'w3_duration', 'w3_num_likes', 'w3_num_comments', 'w3_watched_time', 'w3_like', \
        'w4_duration', 'w4_num_likes', 'w4_num_comments', 'w4_watched_time', 'w4_like', \
        'w5_duration', 'w5_num_likes', 'w5_num_comments', 'w5_watched_time', 'w5_like', \
        'w6_duration', 'w6_num_likes', 'w6_num_comments', 'w6_watched_time', 'w6_like', \
        'w7_duration', 'w7_num_likes', 'w7_num_comments', 'w7_watched_time', 'w7_like', \
        'w8_duration', 'w8_num_likes', 'w8_num_comments', 'w8_watched_time', 'w8_like', \
        'w9_duration', 'w9_num_likes', 'w9_num_comments', 'w9_watched_time', 'w9_like', \
        'wnum_watched_videos_duration', 'wnum_watched_videos_num_likes', 'wnum_watched_videos_num_comments', 'wnum_watched_videos_watched_time', 'wnum_watched_videos_like', \
        'c1_duration', 'c1_num_likes', 'c1_num_comments', \
        'c2_duration', 'c2_num_likes', 'c2_num_comments', \
        'c3_duration', 'c3_num_likes', 'c3_num_comments', \
        'c4_duration', 'c4_num_likes', 'c4_num_comments', \
        't1_duration', 't1_num_likes', 't1_num_comments', \
        'p_like', 'p_has_next', 'p_effective_view'
    ])
    input_row = pd.DataFrame(columns=[
                'uid', \
                'w1_duration', 'w1_num_likes', 'w1_num_comments', 'w1_watched_time', 'w1_like', \
                'w2_duration', 'w2_num_likes', 'w2_num_comments', 'w2_watched_time', 'w2_like', \
                'w3_duration', 'w3_num_likes', 'w3_num_comments', 'w3_watched_time', 'w3_like', \
                'w4_duration', 'w4_num_likes', 'w4_num_comments', 'w4_watched_time', 'w4_like', \
                'w5_duration', 'w5_num_likes', 'w5_num_comments', 'w5_watched_time', 'w5_like', \
                'w6_duration', 'w6_num_likes', 'w6_num_comments', 'w6_watched_time', 'w6_like', \
                'w7_duration', 'w7_num_likes', 'w7_num_comments', 'w7_watched_time', 'w7_like', \
                'w8_duration', 'w8_num_likes', 'w8_num_comments', 'w8_watched_time', 'w8_like', \
                'w9_duration', 'w9_num_likes', 'w9_num_comments', 'w9_watched_time', 'w9_like', \
                'wnum_watched_videos_duration', 'wnum_watched_videos_num_likes', 'wnum_watched_videos_num_comments', 'wnum_watched_videos_watched_time', 'wnum_watched_videos_like', \
                'c1_duration', 'c1_num_likes', 'c1_num_comments', \
                'c2_duration', 'c2_num_likes', 'c2_num_comments', \
                'c3_duration', 'c3_num_likes', 'c3_num_comments', \
                'c4_duration', 'c4_num_likes', 'c4_num_comments', \
                't1_duration', 't1_num_likes', 't1_num_comments', \
                'p_like', 'p_has_next', 'p_effective_view'
            ])

    for file in cleaned_files:
        f_name = file.replace(f'{args.cleaned}\\', '')
        df = pd.read_csv(file, encoding='utf-8')
        slide_time = (df.shape[0] >= (num_watched_videos+num_candidates))*(1 + ((df.shape[0]-(num_watched_videos+num_candidates)) // 5))
        if slide_time < 1:
            print(f'The number of watched videos in {f_name} is less than {num_watched_videos+num_candidates}!!!')
            continue
        for i in range(slide_time):
            # uid
            input_row['uid'] = df.iloc[[0]]['user']
            # watched list
            for j in range(num_watched_videos):
                input_row[f'w{j+1}_duration'] = df.iloc[i*5+j]['video_time']
                input_row[f'w{j+1}_num_likes'] = df.iloc[i*5+j]['like_count']
                input_row[f'w{j+1}_num_comments'] = df.iloc[i*5+j]['comment_count']
                input_row[f'w{j+1}_watched_time'] = df.iloc[i*5+j]['watched_time']
                input_row[f'w{j+1}_like'] = df.iloc[i*5+j]['like']
            
            for t in range(num_recommend):
                # ordered candidates
                for k in range(num_ord_candidates):
                    input_row[f'c{k+1}_duration'] = df.iloc[i*5+num_watched_videos+k]['video_time'] * (t > k)
                    input_row[f'c{k+1}_num_likes'] = df.iloc[i*5+num_watched_videos+k]['like_count'] * (t > k)
                    input_row[f'c{k+1}_num_comments'] = df.iloc[i*5+num_watched_videos+k]['comment_count'] * (t > k)

                for l in range(7-t):
                    # target video
                    input_row['t1_duration'] = df.iloc[i*5+num_watched_videos+l]['video_time']
                    input_row['t1_num_likes'] = df.iloc[i*5+num_watched_videos+l]['like_count']
                    input_row['t1_num_comments'] = df.iloc[i*5+num_watched_videos+l]['comment_count']
                    # output
                    input_row['p_like'] = df.iloc[i*5+num_watched_videos+l]['like']
                    input_row['p_has_next'] = (i*5+num_watched_videos+l != df.shape[0]-3) * 1
                    input_row['p_effective_view'] = (df.iloc[i*5+num_watched_videos+l]['watched_time'] > 5) * 1

                    input_df = pd.concat([input_df, input_row], axis=0, ignore_index=True)
        input_df.to_csv(f'{args.input}/input_{f_name}', index=False)

    # Merge input files
    input_files = glob.glob(f'{args.input}/*.csv')
    merged_df = pd.DataFrame(columns=[
        'uid', \
        'w1_duration', 'w1_num_likes', 'w1_num_comments', 'w1_watched_time', 'w1_like', \
        'w2_duration', 'w2_num_likes', 'w2_num_comments', 'w2_watched_time', 'w2_like', \
        'w3_duration', 'w3_num_likes', 'w3_num_comments', 'w3_watched_time', 'w3_like', \
        'w4_duration', 'w4_num_likes', 'w4_num_comments', 'w4_watched_time', 'w4_like', \
        'w5_duration', 'w5_num_likes', 'w5_num_comments', 'w5_watched_time', 'w5_like', \
        'w6_duration', 'w6_num_likes', 'w6_num_comments', 'w6_watched_time', 'w6_like', \
        'w7_duration', 'w7_num_likes', 'w7_num_comments', 'w7_watched_time', 'w7_like', \
        'w8_duration', 'w8_num_likes', 'w8_num_comments', 'w8_watched_time', 'w8_like', \
        'w9_duration', 'w9_num_likes', 'w9_num_comments', 'w9_watched_time', 'w9_like', \
        'wnum_watched_videos_duration', 'wnum_watched_videos_num_likes', 'wnum_watched_videos_num_comments', 'wnum_watched_videos_watched_time', 'wnum_watched_videos_like', \
        'c1_duration', 'c1_num_likes', 'c1_num_comments', \
        'c2_duration', 'c2_num_likes', 'c2_num_comments', \
        'c3_duration', 'c3_num_likes', 'c3_num_comments', \
        'c4_duration', 'c4_num_likes', 'c4_num_comments', \
        't1_duration', 't1_num_likes', 't1_num_comments', \
        'p_like', 'p_has_next', 'p_effective_view'
    ])

    for file in input_files:
        input_df = pd.read_csv(file, encoding='utf-8')
        merged_df = pd.concat([merged_df, input_df])

    if not for_server:
        merged_df.to_csv(f'{args.input}/final.csv', index=False)
    else:
        merged_df.to_csv(f'{args.input}/server_final.csv', index=False)


do(10)
if args.for_server:
    do(30, args.for_server)