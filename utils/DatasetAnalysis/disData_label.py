import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 高斯函数，用于先增后减趋势，增加偏置项 d
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))  # 只使用3个参数

# 多项式拟合（线性或二次等），增加偏置项
def fit_polynomial_trend(distance_to_end, degree=2):
    positions = np.arange(len(distance_to_end)).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(positions)
    
    model = LinearRegression()
    model.fit(X_poly, distance_to_end)
    distance_pred = model.predict(X_poly)
    
    mse = mean_squared_error(distance_to_end, distance_pred)
    
    return model, distance_pred, mse

# 高斯拟合，增加偏置项
def fit_gaussian_trend(distance_to_end):
    positions = np.arange(len(distance_to_end))
    # 处理零值，使用一个很小的常数替代
    distance_to_end[distance_to_end == 0.0] = 1e-6
    
    # 对数据进行归一化处理
    #positions = np.arange(len(distance_to_end))
    #normalized_distance_to_end = (distance_to_end - np.min(distance_to_end)) / (np.max(distance_to_end) - np.min(distance_to_end))

    # 只使用3个参数进行拟合，去掉偏置项 d
    initial_guess = [max(distance_to_end), np.mean(positions), 1]
    
    # 拟合高斯曲线
    try:
        params, _ = curve_fit(gaussian, positions, distance_to_end, p0=initial_guess)
        fitted_curve = gaussian(positions, *params)
    except RuntimeError as e:
        #print("拟合失败:", e)
        fitted_curve = None
    
    return fitted_curve

# 根据拟合的曲线计算趋势标签
def calculate_trend_label_from_fitted_curve(distance_to_end, predicted_distance, traj_id, fitted_curve=None):
    # 计算拟合曲线的导数（即变化率）
    derivative = np.diff(predicted_distance)
    
    # 判断趋势
    increasing_count = np.sum(derivative > 0)
    decreasing_count = np.sum(derivative < 0)
    # 如果拟合误差较大，判定为不规则趋势
    mse = mean_squared_error(distance_to_end, predicted_distance)
    
    #if mse > 100000:  # 自定义一个阈值（根据数据具体情况调整）
        #print("large mse:", traj_id)
        #return 'irregular'
    
    if increasing_count == len(derivative):
        return 'increasing'  # 递增趋势
    elif decreasing_count == len(derivative):
        return 'decreasing'  # 递减趋势
        #and np.all(fitted_curve == predicted_distance)
    elif fitted_curve is not None:
        # 先增后减趋势通过高斯拟合判断
        return 'increasing_then_decreasing'  # 先增后减
    else:
        return 'irregular'  # 不规则趋势

# 为每条路线计算趋势标签
def get_trend_labels(df_sorted):
    trend_labels = []
    for traj_id, group in df_sorted.groupby('trajID'):
        distance_to_end = group['distance_to_end'].values

        # 如果轨迹长度大于3，则排除最后一个0.0数据
        if len(distance_to_end) > 3 and distance_to_end[-1] == 0.0:
            distance_to_end_new = distance_to_end[:-1]
        else:
            distance_to_end_new = distance_to_end
        
        # 1、对于完整轨迹数据进行处理
        # 先尝试进行多项式拟合，拟合度数为2
        model, predicted_distance, mse = fit_polynomial_trend(distance_to_end, degree=2)
        
        # 如果拟合结果显示是单调递增或者递减，则标记为递增或递减
        trend_label = calculate_trend_label_from_fitted_curve(distance_to_end, predicted_distance, traj_id)
        
        # 如果是单调递增或递减，直接返回；否则尝试进行高斯拟合判断是否是先增后减
        if trend_label == 'irregular':
            fitted_curve = fit_gaussian_trend(distance_to_end)
            trend_label = calculate_trend_label_from_fitted_curve(distance_to_end, predicted_distance, traj_id, fitted_curve=fitted_curve)
        # 2、对于去除最后一个0数据的轨迹进行处理
        _, predicted_distance_new, _ = fit_polynomial_trend(distance_to_end_new, degree=2)
        
        trend_label_new = calculate_trend_label_from_fitted_curve(distance_to_end_new, predicted_distance_new, traj_id)
        
        if trend_label_new == 'irregular':
            fitted_curve_new = fit_gaussian_trend(distance_to_end_new)
            trend_label_new = calculate_trend_label_from_fitted_curve(distance_to_end_new, predicted_distance_new, traj_id, fitted_curve=fitted_curve_new)

        if (trend_label_new != trend_label):
            print("traj_ID:", traj_id, "original label:", trend_label, "new label:", trend_label_new)
        trend_labels.append((traj_id, trend_label))
    
    return trend_labels

# 假设数据集已经加载到 DataFrame df 中
df = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/trajectory_distances_TKY_split200_temp.csv')

# Step 1: 按照 trajID 分组，并根据 POI_position_in_trajectory 排序
df_sorted = df.sort_values(by=['trajID', 'POI_position_in_trajectory'])

# Step 2: 计算趋势标签

# 创建一个空的列来存储趋势标签
df_sorted['trend_label'] = None
# 应用趋势标签
trend_labels = get_trend_labels(df_sorted)

# 将趋势标签添加到数据集中
df_sorted['trend_label'] = df_sorted['trajID'].map(dict(trend_labels))

# Step 3: 保存打上标签后的数据集
df_sorted.to_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/trajectory_distances_TKY_split200_test.csv', index=False)

# Step 4: 可视化分析并保存图像
# 可视化拟合结果
for traj_id, group in df_sorted.groupby('trajID'):
    distance_to_end = group['distance_to_end'].values
    positions = np.arange(len(distance_to_end))
    
    # 进行多项式拟合，拟合度数为2
    model, predicted_distance, mse = fit_polynomial_trend(distance_to_end, degree=2)
    
    # 如果是先增后减趋势，则用高斯拟合
    if 'increasing_then_decreasing' in df_sorted.loc[df_sorted['trajID'] == traj_id, 'trend_label'].values:
        fitted_curve = fit_gaussian_trend(distance_to_end)
        plt.plot(positions, distance_to_end, label=f"Original {traj_id}")
        plt.plot(positions, fitted_curve, label=f"Fitted (Gaussian) {traj_id}")
    else:
        plt.plot(positions, distance_to_end, label=f"Original {traj_id}")
        plt.plot(positions, predicted_distance, label=f"Fitted (Poly) {traj_id}")
    
    plt.title(f"Fitted Curve for Traj {traj_id} (MSE: {mse:.2f})")
    plt.xlabel('POI Position')
    plt.ylabel('Distance to End')
    plt.legend()
    plt.savefig(f"/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/curveFit-TKY/traj_{traj_id}_fitted_curve.png")
    plt.close()
'''
# 保存趋势标签分布图
sns.countplot(x="trend_label", data=df_sorted)
plt.title('Distribution of Trend Labels')
plt.savefig('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/curveFit-Glas/trend_label_distribution.png')  # 保存为 PNG 格式
plt.close()  # 关闭当前图表

# 保存起始时间与趋势标签的关系图
sns.boxplot(x="trend_label", y="startTime", data=df_sorted)
plt.title('Start Time Distribution by Trend Label')
plt.savefig('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/curveFit-Glas/start_time_distribution_by_trend_label.png')
plt.close()

# 保存距离终点与趋势标签的关系图
sns.boxplot(x="trend_label", y="distance_to_end", data=df_sorted)
plt.title('Distance to End by Trend Label')
plt.savefig('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/curveFit-Glas/distance_to_end_by_trend_label.png')
plt.close()

# 保存起始时间和距离终点的散点图
sns.scatterplot(x="startTime", y="distance_to_end", hue="trend_label", data=df_sorted)
plt.title('Scatter plot of Distance to End vs Start Time')
plt.savefig('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/curveFit-Glas/scatter_distance_to_end_vs_start_time.png')
plt.close()

# 保存POI类别与趋势标签的关系图
sns.countplot(x="poiCat", hue="trend_label", data=df_sorted)
plt.title('POI Category vs Trend Label')
plt.savefig('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/curveFit-Glas/poi_category_vs_trend_label.png')
plt.close()
'''

'''
# 加载数据集
poi_data = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/origin_data/poi-Toro.csv')
trajectory_data = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/origin_data/traj-Toro.csv')
distance_data = pd.read_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/trajectory_distances_Toro.csv')

# 将时间戳转换为可读时间格式
trajectory_data['startTime'] = pd.to_datetime(trajectory_data['startTime'], unit='s')

# 合并POI类别
merged_data = pd.merge(distance_data, poi_data[['poiID', 'poiCat']], on='poiID', how='left')

# 合并轨迹长度、访问POI时间
merged_data = pd.merge(merged_data, trajectory_data[['trajID', 'poiID', 'startTime', 'trajLen']], on=['trajID', 'poiID'], how='left')

# 保存合并后的数据集
merged_data.to_csv('/home/lengxiaoting/MyResearch/mamba-main/dataset_new/dataAnaly/disAnaly/trajectory_distances_Toro_temp.csv', index=False)
'''