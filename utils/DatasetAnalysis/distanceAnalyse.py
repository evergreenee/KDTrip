#如果计算相对距离，就可以采样多条轨迹并显示在一张图上了
#改进嵌入部分的POI distance嵌入，

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
'''
#获取数据集中POI据终点距离与POI在序列中的位置
# load dataset
POI_df = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/origin_data/poi-TKY_split200.csv')


trajectory_df = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/origin_data/traj-TKY_split200-order.csv')

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

trajectory_df = trajectory_df.merge(POI_df, left_on='poiID', right_on='poiID', how='left')

valid_trajectory_df = trajectory_df.groupby('trajID').filter(lambda x: len(x) > 3)#>=3
sampled_trajectory_ids = valid_trajectory_df['trajID'].drop_duplicates().sample(n=12, random_state=42)
sampled_trajectory_df = valid_trajectory_df[valid_trajectory_df['trajID'].isin(sampled_trajectory_ids)]

def get_distances_for_trajectory(trajectory_df):
    distances = []
    
    for trajectory_id, group in trajectory_df.groupby('trajID'):
        end_poi = group.iloc[-1] 
        end_lat = end_poi['poiLat']
        end_lon = end_poi['poiLon']
        
        group['distance_to_end'] = group.apply(
            lambda row: calculate_distance(row['poiLat'], row['poiLon'], end_lat, end_lon), axis=1
        )
        
        distances.append(group[['trajID', 'poiID', 'distance_to_end']])
    
    return pd.concat(distances)

distances_df = get_distances_for_trajectory(sampled_trajectory_df)
distances_df['POI_position_in_trajectory'] = distances_df.groupby('trajID').cumcount() + 1

print(distances_df)
distances_df.to_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/trajectory_distances_TKY_split200.csv', index=False)


# draw picture
n_trajectories = len(distances_df['trajID'].unique())  
n_cols = 3  
n_rows = (n_trajectories // n_cols) + (1 if n_trajectories % n_cols > 0 else 0)  

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))  
axes = axes.flatten()  

# 遍历每个轨迹并绘制到相应的子图
for i, (trajectory_id, group) in enumerate(distances_df.groupby('trajID')):
    ax = axes[i]  # 获取当前子图
    ax.plot(group['POI_position_in_trajectory'], group['distance_to_end'], marker='o')  # 绘制轨迹
    ax.set_title(f'Trajectory {trajectory_id}')  # 设置子图标题
    ax.set_xlabel('POI Position in Trajectory')  # x轴标签
    ax.set_ylabel('Distance to End POI (m)')  # y轴标签
    ax.grid(True)  # 显示网格


# 调整布局，防止子图重叠
plt.tight_layout()
plt.savefig('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/trajectory_distance_Glas.png', bbox_inches='tight')
# 显示图形
plt.show()
'''

# 在trajectory-distance数据集中加上POI类别、访问时间等特征
# 加载数据集
poi_data = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/origin_data/poi-TKY_split200.csv')
trajectory_data = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/origin_data/traj-TKY_split200.csv')
distance_data = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/trajectory_distances_TKY_split200_total.csv')

# 将时间戳转换为可读时间格式
trajectory_data['startTime'] = pd.to_datetime(trajectory_data['startTime'], unit='s')

# 合并POI类别
merged_data = pd.merge(distance_data, poi_data[['poiID', 'poiCat']], on='poiID', how='left')

# 合并轨迹长度、访问POI时间
merged_data = pd.merge(merged_data, trajectory_data[['trajID', 'poiID', 'startTime', 'trajLen']], on=['trajID', 'poiID'], how='left')

# 保存合并后的数据集
merged_data.to_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/trajectory_distances_TKY_split200_temp.csv', index=False)


