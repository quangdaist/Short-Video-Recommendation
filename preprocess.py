# python preprocess.py -r ./dataset/raw -c ./dataset/cleaned -i ./dataset/final_input -s

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

    # Create input files from cleaned files
    cleaned_files = glob.glob(f'{args.cleaned}/cleaned_*.csv')

    for file in cleaned_files:
        input_df = pd.DataFrame(columns=['uid', \
            *[f'w{j}_{i}' for j in range(1, num_watched_videos+1) for i in ['duration', 'num_likes', 'num_comments', 'watched_time', 'like'] ], \
            *[f'c{j}_{i}' for j in range(1, num_ord_candidates+1) for i in ['duration', 'num_likes', 'num_comments'] ], \
            't1_duration', 't1_num_likes', 't1_num_comments', \
            'p_like', 'p_has_next', 'p_effective_view'#, 'target_url' 
            ])
        input_row = pd.DataFrame(columns=['uid', \
            *[f'w{j}_{i}' for j in range(1, num_watched_videos+1) for i in ['duration', 'num_likes', 'num_comments', 'watched_time', 'like'] ], \
            *[f'c{j}_{i}' for j in range(1, num_ord_candidates+1) for i in ['duration', 'num_likes', 'num_comments'] ], \
            't1_duration', 't1_num_likes', 't1_num_comments', \
            'p_like', 'p_has_next', 'p_effective_view'#, 'target_url'   
            ])
            
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
            
            for t in range(num_candidates):
                # ordered candidates
                for k in range(num_ord_candidates):
                    input_row[f'c{k+1}_duration'] = df.iloc[i*5+num_watched_videos+k+(t%num_ord_candidates)*(t>num_ord_candidates)]['video_time'] * (t > k)
                    input_row[f'c{k+1}_num_likes'] = df.iloc[i*5+num_watched_videos+k+(t%num_ord_candidates)*(t>num_ord_candidates)]['like_count'] * (t > k)
                    input_row[f'c{k+1}_num_comments'] = df.iloc[i*5+num_watched_videos+k+(t%num_ord_candidates)*(t>num_ord_candidates)]['comment_count'] * (t > k)

                # target video
                input_row['t1_duration'] = df.iloc[i*5+num_watched_videos+t]['video_time']
                input_row['t1_num_likes'] = df.iloc[i*5+num_watched_videos+t]['like_count']
                input_row['t1_num_comments'] = df.iloc[i*5+num_watched_videos+t]['comment_count']
                #input_row['target_url'] = df.iloc[i*5+num_watched_videos+t]['url']
                # output
                input_row['p_like'] = df.iloc[i*5+num_watched_videos+t]['like']
                input_row['p_has_next'] = (i*5+num_watched_videos+t != df.shape[0]-3) * 1
                input_row['p_effective_view'] = (df.iloc[i*5+num_watched_videos+t]['watched_time'] > 5) * 1

                input_df = pd.concat([input_df, input_row], axis=0, ignore_index=True)
                # t=0: -> 0              -> t1=video11
                # t=1: k=0 -> c1=video11 -> t1=video12
                # t=2: k=0 -> c1=video11
                #      k=1 -> c2=video12 -> t1=video13
                # t=3: k=0 -> c1=video11
                #      k=1 -> c2=video12
                #      k=2 -> c3=video13 -> t1=video14
                # t=4: k=0 -> c1=video11
                #      k=1 -> c2=video12
                #      k=2 -> c3=video13
                #      k=3 -> c4=video14 -> t1=video15
                # t=5: k=0 -> c1=video12
                #      k=1 -> c2=video13
                #      k=2 -> c3=video14
                #      k=3 -> c4=video15 -> t1=video16
                # t=6: k=0 -> c1=video13
                #      k=1 -> c2=video14
                #      k=2 -> c3=video15
                #      k=3 -> c4=video16 -> t1=video17

        idx = (int(input_df.shape[0] * 0.8) // 7) * 7
        train_df = input_df.iloc[:idx]
        test_df = input_df.iloc[idx:]
        if not for_server:
            train_df.to_csv(f'{args.input}/train_{f_name}', index=False)
            test_df.to_csv(f'{args.input}/test_{f_name}', index=False)
        else:
            train_df.to_csv(f'{args.input}/server_train_{f_name}', index=False)
            test_df.to_csv(f'{args.input}/server_test_{f_name}', index=False)

    
    if not for_server:
        train_files = glob.glob(f'{args.input}/train_*.csv')
        test_files = glob.glob(f'{args.input}/test_*.csv')
    else:
        train_files = glob.glob(f'{args.input}/server_train_*.csv')
        test_files = glob.glob(f'{args.input}/server_test_*.csv')
    # Merge train files
    merged_train_df = pd.DataFrame(columns=['uid', \
        *[f'w{j}_{i}' for j in range(1, num_watched_videos+1) for i in ['duration', 'num_likes', 'num_comments', 'watched_time', 'like'] ], \
        *[f'c{j}_{i}' for j in range(1, num_ord_candidates+1) for i in ['duration', 'num_likes', 'num_comments'] ], \
        't1_duration', 't1_num_likes', 't1_num_comments', \
        'p_like', 'p_has_next', 'p_effective_view'#, 'target_url'  
        ])

    for file in train_files:
        input_df = pd.read_csv(file, encoding='utf-8')
        merged_train_df = pd.concat([merged_train_df, input_df])

    if not for_server:
        merged_train_df.to_csv(f'{args.input}/final_train.csv', index=False)
    else:
        merged_train_df.to_csv(f'{args.input}/server_final_train.csv', index=False)
    
    # Merge test files
    merged_test_df = pd.DataFrame(columns=['uid', \
        *[f'w{j}_{i}' for j in range(1, num_watched_videos+1) for i in ['duration', 'num_likes', 'num_comments', 'watched_time', 'like'] ], \
        *[f'c{j}_{i}' for j in range(1, num_ord_candidates+1) for i in ['duration', 'num_likes', 'num_comments'] ], \
        't1_duration', 't1_num_likes', 't1_num_comments', \
        'p_like', 'p_has_next', 'p_effective_view'#, 'target_url'  
        ])

    for file in test_files:
        input_df = pd.read_csv(file, encoding='utf-8')
        merged_test_df = pd.concat([merged_test_df, input_df])

    if not for_server:
        merged_test_df.to_csv(f'{args.input}/final_test.csv', index=False)
    else:
        merged_test_df.to_csv(f'{args.input}/server_final_test.csv', index=False)


do(10)
if args.for_server:
    do(30, args.for_server)