

"""Goal see if data from ___ is consistent

basic structure of what I'mma do:

Preprocessing
run thru all the years I have data for
    turn them into dataframes
    only need stuff from metropolitan police
    only streets?

combine all data frames
sort by month/year?

write a stacked line chart for all types of crime over the years
    one for everywhere
    one for each ward? idk if this is usefull, but I'll need to map LSOAs to wards and that seems annoying
overlay important events??

"""
import os
import pandas as pd

base_dir = r'C:\Users\mgshe\PycharmProjects\Addressing-real-world-crime-and-security-problems-with-data-science\Data\All_data'

# Filters
jurisdiction = 'metropolitan'
datasets = {'outcomes', 'stop-and-search', 'street'}
datasets = {'street'}

# combine me up here
all_dfs = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.csv'):
            # Example filename: 2016-02-metropolitan-street.csv
            name_parts = file[:-4].lower().split('-')  # Remove .csv and split
            if len(name_parts) >= 4:
                # Dataset name could have multiple parts (like stop-and-search)
                year, month, jur, *dataset_parts = name_parts
                dataset_name = '-'.join(dataset_parts)
                if jur == jurisdiction and dataset_name in datasets:
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        df['source_file'] = file  # Optional: track file origin
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

# combine and gimme 1 file I can work with
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    output_path = r'combined_streets_data.csv'
    combined_df.to_csv(output_path, index=False)
    print(f'Combined data saved to {output_path}')
else:
    print('No matching CSV files found.')

