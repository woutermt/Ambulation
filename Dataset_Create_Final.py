"""
Wouter Tijs
Orikami, Nijmegen
06-09-2018
"""

from HiddenDatabaseURI import mongo_uri
from pymongo import MongoClient
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
import pandas as pd
import numpy as np


# VARIABLES
dir_datasets = './Intermediate_results/'
dir_subsets = './Intermediate_results/Pooled_subsets/'
dir_imgs = './Intermediate_results/Imgs/'
question_id = 'qxR2LdLvytFtLNsNW'                   # Database question id

columns_to_binary = ['questionScore', 'steps']      # Two columns to be scored 1/0 split on mean,
                                                    # and possibly pooled to 1-4 or 1-9 label set

method = 'multi'            # Method of processing:
                                # 'multi':  create dataset
                                # 'freq':   create plots with response counts
req_entries = 5             # Minimum number of required measurements for entry into dataset
subset = None               # Optional subsetting of measurements for specific interval in time
num_class = 2               # Number of splits we apply to question score class
thres_worn = 1440 * 5/6     # Threshold for wear time to add data


# Users to be removed (incompatible with fetching from database
id_remove = ['ykL6Ce8XmPoiqyAMW',       # B19: 0 Activity
             'S8gTWpiMcgjKXNCXy',       # A15: 1 Ans + Activity
             'Dzk7fSkuN94xGo2Nn',       # B05: 0 Ans
             'uqJxzHCvFkLdtbqNB',       # B24: 0 Ans
             'qkvvDjL5xH4QX25n2',       # B03: 0 Ans
             'ejRHjXD7gkkqPqt33',       # A13: Super inconsistent, 2x as many average steps
             'KtyoaZvPTmc8Tj3JG',       # Dropout
             '4ktCwsYwJ74tfAhz9']       # Dropout
             # '5jzciJ9n93h9W5sPR']       # A22: 8/24 Q-score non 0, rest < 1


def write_datasets(ms_all, hc_all):

    ms_all.to_csv(dir_subsets + 'MS_min5_bin.csv')
    hc_all.to_csv(dir_subsets + 'HC_min5_bin.csv')


def count_response_freq(user_data, tot_ans):

    if num_class == 3:
        bin_options = [0, 2, 3, 4, 6, 10, 11, 14, 15]

    elif num_class == 2:
        bin_options = [0, 1, 2, 3]

    # Counting occurrences
    count = user_data['freq_response'].value_counts()

    # Missing values set to 0
    for i in bin_options:
        if i not in count.index:
            count[i] = 0

    # Matrix of counts
    if num_class == 3:
        count_matrix = np.matrix([[count[6]/tot_ans, count[14]/tot_ans, count[15]/tot_ans],
                                  [count[4]/tot_ans, count[10]/tot_ans, count[11]/tot_ans],
                                  [count[0]/tot_ans, count[2]/tot_ans, count[3]/tot_ans]])

    elif num_class == 2:
        count_matrix = np.matrix([[count[2]/tot_ans, count[3]/tot_ans],
                                  [count[0]/tot_ans, count[1]/tot_ans]])

    count_matrix = count_matrix.round(decimals=2)

    return count_matrix


def plot_frequency(count_matrix, bins, user_code, tot_ans):

    # Plotting
    plt.figure()
    plt.imshow(count_matrix, cmap='Greens', vmax=1, vmin=0)
    plt.colorbar()
    plt.title(user_code + ' Tot ans: ' + str(tot_ans) + ' Splits: '+
              str(bins[0]) + '/' + str(bins[1]) + '/' + str(bins[2]) + ' and ' +
              str(bins[3]) + '/' + str(bins[4]) + '/' + str(bins[5]) + '\n' +
              str(count_matrix.tolist()))
    plt.xlabel('bad day                 av.                 good day')
    plt.ylabel('low activity            av.            high activity')
    plt.xticks([])
    plt.yticks([])

    plt.show()
    # plt.savefig(dir_imgs + 'One_std_norm/' + 'mean ' + user_code + '.png')


def binary_features(user_data, num_class):

    splits = []

    # Convert columns to binary of average
    for col in columns_to_binary:
        num_a, num_b, num_c = 0, 0, 0

        col_split_val = user_data[col].mean()
        stdv = user_data[col].std()

        if num_class == 3:

            split_lower = col_split_val - .5 * stdv
            split_upper = col_split_val + .5 * stdv

            for idx, value in user_data[col].iteritems():
                if value > split_upper:
                    user_data.at[idx, col + 'Binary'] = '11'
                    num_a += 1
                if value < split_lower:
                    user_data.at[idx, col + 'Binary'] = '0'
                    num_c += 1
                if split_lower < value < split_upper:
                    user_data.at[idx, col + 'Binary'] = '10'
                    num_b += 1

        elif num_class == 2:

            for idx, value in user_data[col].iteritems():
                if value >= col_split_val:
                    user_data.at[idx, col + 'Binary'] = 1
                    num_a += 1
                else:
                    user_data.at[idx, col + 'Binary'] = 0
                    num_b += 1

        splits += [num_a, num_b, num_c]

    # Pooling binary steps and questionScore into 4 classes
    freq_response_binary = user_data[columns_to_binary[1]+'Binary'].map(int).map(str) + \
                           user_data[columns_to_binary[0]+'Binary'].map(int).map(str)

    user_data['freq_response'] = freq_response_binary.apply(int, args=(2,))

    return user_data, splits


def process_data(user_data, user_id, user_info):

    # Add column with ids, codes and other user variables:
    # weight (kg), length (cm), gender (m/f), mean TUG score (sec)
    user_data['id'] = user_id
    user_data['code'] = user_info.at[user_id, 'code']
    user_data['weight'] = user_info.at[user_id, 'weight']
    user_data['length'] = user_info.at[user_id, 'length']
    user_data['gender'] = user_info.at[user_id, 'gender']

    mean_tug = user_info[['tug1', 'tug2']].mean(axis=1)
    user_data['tug'] = mean_tug[user_id]

    # Dropping any columns lacking heartRateZones
    # (only relevant when the answer_data is occluded)
    user_data = user_data.dropna(subset=['heartRateZones'])

    # Count mins of heart rate registration
    min_worn = []
    for entry in user_data['heartRateZones']:
        min_worn.append(sum(json_normalize(entry)['minutes']))
    user_data['minWorn'] = min_worn

    # Filtering out short wear times
    user_data = user_data[user_data['minWorn'] >= thres_worn]

    # Drop unneeded cols
    user_data = user_data.drop(
        columns=['activeScore', 'distances', 'heartRateZones', 'caloriesBMR'])

    # users with customHeartRateZones, doesn't work for duplicate removal
    if user_id == 'nTzsjwEMxWdgxFWjK' or user_id == 'ScycT3hXoqEbhNnaH':
        user_data = user_data.drop(columns=['customHeartRateZones'])

    # Remove duplicates
    user_data = user_data.drop_duplicates()

    # Sort values to get measurements chronologically
    user_data = user_data.sort_values('timestamp')

    return user_data


def query_activity(db_fit, user_id):

    # Retrieve database activity data
    dailyfit_data = db_fit.fitBitDataScriptFeb2017.find(
        {'docId': user_id, 'name': 'dailysummary'},         # Data conditions
        {'_id': 0, 'timestamp': 1, 'value': 1}              # Requested data
    )

    # Unpacking embedded columns from 'value'
    dailyfit_data = pd.DataFrame(json_normalize(list(dailyfit_data)))

    # Removing 'value.' from all unpacked embedded elements in value column
    dailyfit_data = dailyfit_data.rename(
        columns={col: col.split('.')[1] for col in dailyfit_data.columns[1:]})

    # Reformat timestamp to date only for daily character
    dailyfit_data['timestamp'] = pd.to_datetime(dailyfit_data['timestamp']).dt.date

    return dailyfit_data


def query_answers(db_call, user_id):

    # Retrieve database answer data
    answer_data = db_call.observations.find(
        {'userId': user_id, 'questionId': question_id},     # Data conditions
        {'_id': 0, 'timestamp': 1, 'value': 1}              # Requested data
    )

    # Commit data to dataframe
    answer_data = pd.DataFrame(list(answer_data))

    # Reformat timestamp to date only for daily character
    answer_data['timestamp'] = pd.to_datetime(answer_data['timestamp']).dt.date

    # Rename value column for clarity
    answer_data = answer_data.rename(columns={'value': 'questionScore'})

    return answer_data


def main():

    ms_all, hc_all, freq_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Connect to databases
    client = MongoClient(mongo_uri)
    db_fit = client["diapro"]
    db_call = client["mijn-kwik"]

    # Retrieve user data
    user_info = db_call.userCodes.find(
        {},                                                 # Data conditions
        {'_id': 0, 'userId': 1, 'code': 1, 'length': 1,     # Requested data
         'weight': 1, 'gender': 1, 'tug1': 1, 'tug2': 1}    # Requested data
    )

    user_info = pd.DataFrame(list(user_info))

    # Remove unwanted users
    user_info = user_info.set_index('userId').drop(id_remove)

    for user_id in user_info.index:
        user_code = user_info.at[user_id, 'code']

        # Gather data
        dailyfit_data = query_activity(db_fit, user_id)
        answer_data = query_answers(db_call, user_id)

        # Merge and process data
        user_data = pd.merge(dailyfit_data, answer_data, on='timestamp')
        user_data = process_data(user_data, user_id, user_info)

        # Split variables on their mean
        user_data, bin_splits = binary_features(user_data, 2)

        # DATA/USER CONTROL
        # Enforce rules for min. measurements
        if user_data.shape[0] < req_entries:
            print('%s - %s with %i entries skipped' %
                  (user_id, user_code, user_data.shape[0]))
            continue
        else:
            print('%s - %s: Entries reduced to %i'
                  % (user_id, user_info.at[user_id, 'code'], user_data.shape[0]))
            if subset:
                user_data = user_data[:subset]

        # METHOD SELECTION
        # (1) Data collection for statistical analysis

        if method == 'multi':
            # Eval user disease status
            if user_code[0] == 'A':
                ms_all = pd.concat([ms_all, user_data])
            else:
                hc_all = pd.concat([hc_all, user_data])

            # Write output data
            write_datasets(ms_all, hc_all)

        # (2) Counts and behavioral plots
        if method == 'freq':
            # Eval user disease status
            if user_code[0] == 'A':

                # Split variables on their mean
                user_data, bin_splits = binary_features(user_data, num_class)

                # Count and plot
                tot_ans = user_data.shape[0]
                count_matrix = count_response_freq(user_data, tot_ans)
                plot_frequency(count_matrix, bin_splits, user_code, tot_ans)


""" START """
if __name__ == '__main__':
    main()

