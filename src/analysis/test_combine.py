import pandas as pd
import json

# 读取CSV文件
csv_file_path = 'mapped_data_auction.csv'
df_csv = pd.read_csv(csv_file_path)

# 读取JSON文件
json_file_path = 'survey_number_rounds_clean.json'
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# 将JSON数据转换为适合合并的格式
# 我们创建一个新列 'highest increment in bidding' 并初始化为默认值（例如0或NaN）
# df_csv['highest increment in bidding'] = 0  # 或者使用 pd.NA 代替 0

# 遍历JSON数据并更新DataFrame
for key, value in json_data.items():
    if key.isdigit() and int(key) < len(df_csv):
        df_csv.at[int(key), 'number_rounds'] = value 
        # if value == 1:
        #     df_csv.at[int(key), 'highest_win_bid'] = 1
        # else:
        #     df_csv.at[int(key), 'highest_win_bid'] = 0

# 写入新的CSV文件
df_csv.to_csv('mapped_data_auction.csv', index=False)

# import pandas as pd

# # Read the CSV file
# csv_file_path = 'raw_data_bail.csv'
# df_csv = pd.read_csv(csv_file_path)

# # Iterate over the DataFrame and update the 'defendant satisfaction' column
# for index, row in df_csv.iterrows():
#     if row['bail_amt'] > 50000:
#         df_csv.at[index, 'defendant satisfaction'] = 1
#     else:
#         df_csv.at[index, 'defendant satisfaction'] = 3

# # Write the updated DataFrame to a new CSV file
# df_csv.to_csv('new_data.csv', index=False)

