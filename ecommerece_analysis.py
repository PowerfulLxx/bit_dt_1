import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pyarrow.parquet as pq
import dask.dataframe as dd
import os
import json
import warnings
from datetime import datetime
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')

start_time = datetime.now()

# 为matplotlib配置中文字体
def setup_chinese_font():
    # 根据系统中已经存在的中文字体设置
    font_names = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'AR PL UMing CN']
    
    # 检查是否有可用的中文字体
    available_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
    print("系统中的字体:", [f for f in available_fonts if any(x.lower() in f.lower() for x in ['wenquanyi', 'noto', 'uming', 'cjk', 'wqy', '文泉'])])
    
    # 检查指定的字体是否可用
    font_found = False
    chinese_font = None
    
    for font_name in font_names:
        for available_font in available_fonts:
            if font_name.lower() in available_font.lower():
                chinese_font = available_font
                font_found = True
                print(f"使用中文字体: {chinese_font}")
                break
        if font_found:
            break
            
    if not font_found:
        print("警告: 未找到可用的中文字体，尝试使用系统默认字体")
        # 尝试使用可能存在的字体路径
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                chinese_font = FontProperties(fname=path)
                font_found = True
                print(f"使用字体文件: {path}")
                break
    
    return chinese_font

# 获取中文字体
chinese_font = setup_chinese_font()

# 配置绘图时使用的字体
def set_chinese_on_plot():
    if chinese_font:
        plt.rcParams['axes.unicode_minus'] = False

# 修改绘图函数，显式设置字体
def plot_with_chinese(plot_func, title, xlabel=None, ylabel=None, figsize=(10, 6), *args, **kwargs):
    """包装绘图函数，确保中文正确显示"""
    fig = plt.figure(figsize=figsize)
    
    # 执行绘图函数
    plot_func(*args, **kwargs)
    
    # 设置标题和轴标签，明确指定字体
    if title and chinese_font:
        plt.title(title, fontproperties=chinese_font, fontsize=15)
    elif title:
        plt.title(title, fontsize=15)
        
    if xlabel and chinese_font:
        plt.xlabel(xlabel, fontproperties=chinese_font, fontsize=12)
    elif xlabel:
        plt.xlabel(xlabel, fontsize=12)
        
    if ylabel and chinese_font:
        plt.ylabel(ylabel, fontproperties=chinese_font, fontsize=12)
    elif ylabel:
        plt.ylabel(ylabel, fontsize=12)
    
    return fig

print("=" * 50)
print("电商用户行为数据分析报告")
print("=" * 50)

# 1. 数据加载
print("\n1. 数据加载")
print("-" * 50)

# 获取所有parquet文件列表
parquet_files = [f for f in os.listdir('.') if f.startswith('part-') and f.endswith('.parquet')]
print(f"发现 {len(parquet_files)} 个parquet文件: {', '.join(parquet_files)}")

# 使用Dask处理所有文件
print("使用Dask读取所有parquet文件...")
ddf = dd.read_parquet('part-*.parquet')

# 计算总行数
print("计算总行数...")
total_rows = len(ddf)
print(f"数据集总行数（所有文件）: {ddf.shape[0].compute():,}")
print(f"数据集列名: {ddf.columns.tolist()}")

# 创建输出目录
output_dir = 'analysis_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 数据预处理与质量评估
print("\n\n2. 数据预处理与质量评估")
print("-" * 50)

# 2.1 检查缺失值 - 使用Dask计算
print("\n2.1 检查缺失值")
print("计算缺失值统计（这可能需要一些时间）...")
missing_values = ddf.isnull().sum().compute()
missing_percentage = (missing_values / ddf.shape[0].compute()) * 100
missing_data = pd.concat([missing_values, missing_percentage], axis=1)
missing_data.columns = ['缺失值数量', '缺失百分比(%)']
print(missing_data[missing_data['缺失值数量'] > 0])

if missing_data['缺失值数量'].sum() == 0:
    print("数据中没有缺失值。")
else:
    print(f"发现缺失值，总计: {missing_data['缺失值数量'].sum()}")

# 2.2 检查数据类型
print("\n2.2 数据类型")
print("数据类型:")
print(ddf.dtypes)

# 2.3 检查并处理异常值 - 这部分需要对整个数据集进行计算
print("\n2.3 检查异常值")
print("计算数值列的统计信息（这可能需要一些时间）...")

numeric_cols = ['age', 'income', 'credit_score']
outliers_stats = {}

# 使用Dask计算每列的四分位数和范围
for col in numeric_cols:
    print(f"计算 {col} 列的统计信息...")
    # 计算分位数
    q_df = ddf[col].quantile([0.25, 0.75]).compute()
    q1, q3 = q_df[0.25], q_df[0.75]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 计算最大最小值
    min_val = ddf[col].min().compute()
    max_val = ddf[col].max().compute()
    
    # 计算异常值数量 (这个操作比较耗资源，因此用条件语句分别计算)
    n_lower_outliers = ddf[ddf[col] < lower_bound].shape[0].compute()
    n_upper_outliers = ddf[ddf[col] > upper_bound].shape[0].compute()
    n_outliers = n_lower_outliers + n_upper_outliers
    outlier_rate = (n_outliers / ddf.shape[0].compute()) * 100
    
    print(f"{col} 列异常值数量: {n_outliers}, 占比: {outlier_rate:.2f}%")
    print(f"{col} 合理范围: ({lower_bound:.2f}, {upper_bound:.2f})")
    print(f"{col} 实际范围: ({min_val}, {max_val})")
    
    outliers_stats[col] = {
        'outlier_count': n_outliers,
        'outlier_rate': outlier_rate,
        'reasonable_range': (lower_bound, upper_bound),
        'actual_range': (min_val, max_val)
    }

# 2.4 处理用户购买历史数据
print("\n2.4 处理用户购买历史数据")
# 为了避免处理整个数据集的JSON，我们先抽取少量样本查看结构
# 使用frac参数而不是n参数，因为Dask不支持n参数
sample_for_inspection = ddf.sample(frac=0.0001).compute()
print("购买历史样本（仅供检查格式）：")
sample_purchase = sample_for_inspection['purchase_history']
print(sample_purchase)

# 尝试解析第一条记录，了解结构
try:
    first_record = json.loads(sample_for_inspection['purchase_history'].iloc[0])
    print("\n购买历史是有效的JSON格式。示例结构：")
    print(json.dumps(first_record, indent=2, ensure_ascii=False)[:300] + "...")
    
    # 使用Dask DataFrame添加购买次数列
    
except Exception as e:
    print(f"\n解析购买历史时出错: {e}")
    print("跳过购买历史分析")

# 3. 探索性分析与可视化
print("\n\n3. 探索性分析与可视化")
print("-" * 50)

# 3.1 年龄分布可视化
print("\n3.1 年龄分布可视化")
print("计算年龄分布...")
age_counts = ddf['age'].compute()
set_chinese_on_plot()
fig = plot_with_chinese(
    lambda: sns.histplot(age_counts, bins=20, kde=True),
    title='用户年龄分布',
    xlabel='年龄',
    ylabel='用户数量'
)
plt.savefig(f'{output_dir}/age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"年龄分布图已保存至 {output_dir}/age_distribution.png")

# 3.2 收入分布可视化
print("\n3.2 收入分布可视化")
print("计算收入分布...")
income_counts = ddf['income'].compute()
set_chinese_on_plot()
fig = plot_with_chinese(
    lambda: sns.histplot(income_counts, bins=30, kde=True),
    title='用户收入分布',
    xlabel='收入',
    ylabel='用户数量'
)
plt.ticklabel_format(style='plain', axis='x')
plt.savefig(f'{output_dir}/income_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"收入分布图已保存至 {output_dir}/income_distribution.png")

# 3.3 性别分布饼图
print("\n3.3 性别分布可视化")
print("计算性别分布...")
gender_counts = ddf['gender'].value_counts().compute()
set_chinese_on_plot()
plt.figure(figsize=(8, 8))

# 生成饼图但不包含标签
plt.pie(gender_counts, autopct='%1.1f%%', startangle=90)

# 单独添加标题，使用中文字体
if chinese_font:
    plt.title('用户性别分布', fontproperties=chinese_font, fontsize=15)
else:
    plt.title('用户性别分布', fontsize=15)

# 添加图例，使用中文字体
if chinese_font:
    plt.legend(gender_counts.index, prop=chinese_font)
else:
    plt.legend(gender_counts.index)

plt.axis('equal')
plt.savefig(f'{output_dir}/gender_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"性别分布图已保存至 {output_dir}/gender_distribution.png")

# 3.4 国家分布前10名
print("\n3.4 国家分布可视化")
print("计算国家分布...")
country_counts = ddf['country'].value_counts().compute().head(10)
set_chinese_on_plot()
fig = plot_with_chinese(
    lambda: sns.barplot(x=country_counts.index, y=country_counts.values),
    title='用户国家分布(前10名)',
    xlabel='国家',
    ylabel='用户数量',
    figsize=(12, 6)
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/country_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"国家分布图已保存至 {output_dir}/country_distribution.png")

# 3.5 注册年份趋势
print("\n3.5 用户注册时间趋势")
print("计算注册年份分布...")
# 首先将日期时间列转换为datetime类型
ddf['registration_date'] = dd.to_datetime(ddf['registration_date'])
# 提取年份并计算分布
reg_year = ddf['registration_date'].dt.year
reg_year_counts = reg_year.value_counts().compute().sort_index()

set_chinese_on_plot()
fig = plot_with_chinese(
    lambda: sns.lineplot(x=reg_year_counts.index, y=reg_year_counts.values, marker='o'),
    title='用户注册年份趋势',
    xlabel='年份',
    ylabel='注册用户数',
    figsize=(12, 6)
)
plt.xticks(reg_year_counts.index)
plt.tight_layout()
plt.savefig(f'{output_dir}/registration_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"注册趋势图已保存至 {output_dir}/registration_trend.png")


# 4. 用户画像分析
print("\n\n4. 用户画像分析")
print("-" * 50)

# 4.1 用户活跃度分析
print("\n4.1 用户活跃度分析")
print("计算用户活跃度...")
active_counts_total = ddf['is_active'].value_counts().compute()
active_percent = (active_counts_total.get(True, 0) / active_counts_total.sum()) * 100
inactive_percent = (active_counts_total.get(False, 0) / active_counts_total.sum()) * 100

print(f"活跃用户数: {active_counts_total.get(True, 0):,}, 占比: {active_percent:.2f}%")
print(f"非活跃用户数: {active_counts_total.get(False, 0):,}, 占比: {inactive_percent:.2f}%")

# 4.2 用户聚类分析
print("\n4.2 用户聚类分析")
print("使用批处理方式进行聚类分析...")

# 选择用于聚类的特征
cluster_features = ['age', 'income', 'credit_score']
if 'purchase_count' in ddf.columns:
    cluster_features.append('purchase_count')
    
print("聚类特征:", cluster_features)

try:
    # 使用增量式KMeans进行聚类
    from sklearn.cluster import MiniBatchKMeans
    
    # 设置批处理参数
    batch_size = 10000  # 每批处理的数据量
    n_clusters = 4  # 聚类数量
    
    print(f"使用MiniBatchKMeans进行聚类，聚类数量 = {n_clusters}, 批大小 = {batch_size}")
    
    # 1. 首先使用小量数据进行标准化参数计算
    print("计算标准化参数...")
    sample_for_scaling = ddf[cluster_features].head(50000)
    scaler = StandardScaler()
    scaler.fit(sample_for_scaling)
    print("标准化参数计算完成")
    
    # 2. 初始化MiniBatchKMeans
    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42, n_init=3)
    
    # 3. 分批处理数据
    n_partitions = ddf.npartitions
    total_processed = 0
    valid_processed = 0
    
    print(f"开始批量聚类，总分区数: {n_partitions}")
    
    for i in range(n_partitions):
        try:
            print(f"处理分区 {i+1}/{n_partitions}...")
            # 获取当前分区数据
            partition_data = ddf[cluster_features].get_partition(i).compute()
            # 处理空值
            valid_data = partition_data.dropna()
            
            if len(valid_data) > 0:
                # 更新累计数
                total_processed += len(partition_data)
                valid_processed += len(valid_data)
                
                # 标准化当前批次数据
                scaled_batch = scaler.transform(valid_data)
                
                # 部分拟合当前批次
                mbk.partial_fit(scaled_batch)
                
                # 每5个分区打印一次进度
                if (i + 1) % 5 == 0 or i == n_partitions - 1:
                    print(f"已处理 {total_processed:,} 条记录，有效数据 {valid_processed:,} 条")
            
        except Exception as e:
            print(f"处理分区 {i} 时出错: {e}")
            continue
    
    print("聚类完成！")
    
    # 4. 获取聚类中心并反标准化
    cluster_centers = mbk.cluster_centers_
    cluster_centers_original = scaler.inverse_transform(cluster_centers)
    
    # 创建聚类分析结果DataFrame
    cluster_analysis = pd.DataFrame(cluster_centers_original, 
                                   columns=cluster_features)
    cluster_analysis.index.name = 'cluster'
    
    print("\n各聚类特征分析:")
    print(cluster_analysis)
    
    # 将聚类分析结果可视化
    set_chinese_on_plot()
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(cluster_features):
        plt.subplot(2, 2, i+1)
        sns.barplot(x=cluster_analysis.index, y=cluster_analysis[feature])
        
        # 设置标题和标签，使用中文字体
        if chinese_font:
            plt.title(f'聚类中心 - {feature}', fontproperties=chinese_font, fontsize=13)
            plt.xlabel('聚类', fontproperties=chinese_font, fontsize=11)
            plt.ylabel(feature, fontproperties=chinese_font, fontsize=11)
        else:
            plt.title(f'聚类中心 - {feature}', fontsize=13)
            plt.xlabel('聚类', fontsize=11)
            plt.ylabel(feature, fontsize=11)
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"聚类特征分析图已保存至 {output_dir}/cluster_features.png")
    
    # 5. 计算聚类分布 (再次遍历分区统计每个聚类的数量)
    print("计算聚类分布...")
    cluster_counts = np.zeros(n_clusters, dtype=np.int64)
    
    # 每次处理一个分区，防止内存溢出
    for i in range(n_partitions):
        try:
            partition_data = ddf[cluster_features].get_partition(i).compute()
            valid_data = partition_data.dropna()
            
            if len(valid_data) > 0:
                # 标准化数据
                scaled_data = scaler.transform(valid_data)
                
                # 预测聚类
                batch_labels = mbk.predict(scaled_data)
                
                # 更新计数
                for label in range(n_clusters):
                    cluster_counts[label] += np.sum(batch_labels == label)
                
                # 每10个分区打印一次进度
                if (i + 1) % 10 == 0 or i == n_partitions - 1:
                    print(f"聚类分布计算进度: {i+1}/{n_partitions}")
        except Exception as e:
            print(f"处理分区 {i} 时出错: {e}")
            continue
    
    # 计算聚类百分比
    total_clustered = np.sum(cluster_counts)
    cluster_percentages = (cluster_counts / total_clustered * 100) if total_clustered > 0 else np.zeros(n_clusters)
    
    # 打印聚类分布
    print("\n聚类分布结果:")
    for cluster_id in range(n_clusters):
        count = cluster_counts[cluster_id]
        percentage = cluster_percentages[cluster_id]
        print(f"聚类 {cluster_id}: {count:,} 用户 ({percentage:.2f}%)")
        for j, feature in enumerate(cluster_features):
            mean_value = cluster_centers_original[cluster_id, j]
            print(f"  平均{feature}: {mean_value:.2f}")
    
    # 绘制聚类分布饼图
    set_chinese_on_plot()
    plt.figure(figsize=(10, 10))
    wedges, texts, autotexts = plt.pie(
        cluster_counts, 
        autopct='%1.1f%%', 
        startangle=90, 
        shadow=False
    )
    
    # 设置饼图文字属性
    if chinese_font:
        plt.title('用户聚类分布', fontproperties=chinese_font, fontsize=16)
        # 添加图例
        labels = [f'聚类 {i} ({cluster_counts[i]:,})' for i in range(n_clusters)]
        plt.legend(wedges, labels, title="聚类", prop=chinese_font, 
                   loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    else:
        plt.title('用户聚类分布', fontsize=16)
        labels = [f'聚类 {i} ({cluster_counts[i]:,})' for i in range(n_clusters)]
        plt.legend(wedges, labels, title="聚类", loc="center left", 
                   bbox_to_anchor=(1, 0, 0.5, 1))
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"聚类分布图已保存至 {output_dir}/cluster_distribution.png")
    
    # 6. 使用PCA可视化聚类中心 (只可视化聚类中心，不可视化所有点)
    print("创建PCA可视化...")
    pca = PCA(n_components=2)
    pca.fit(scaler.transform(sample_for_scaling))  # 使用采样数据拟合PCA
    
    # 对聚类中心进行降维
    cluster_centers_2d = pca.transform(cluster_centers)
    
    # 绘制聚类中心
    set_chinese_on_plot()
    plt.figure(figsize=(10, 8))
    plt.scatter(
        cluster_centers_2d[:, 0], 
        cluster_centers_2d[:, 1], 
        c=range(n_clusters), 
        cmap='viridis', 
        s=200, 
        marker='o', 
        edgecolors='black'
    )
    
    # 标记聚类中心编号
    for i, (x, y) in enumerate(cluster_centers_2d):
        plt.text(x, y, str(i), fontsize=16, ha='center', va='center', color='white')
    
    if chinese_font:
        plt.title('PCA降维的聚类中心可视化', fontproperties=chinese_font, fontsize=15)
        plt.xlabel('主成分1', fontproperties=chinese_font, fontsize=12)
        plt.ylabel('主成分2', fontproperties=chinese_font, fontsize=12)
    else:
        plt.title('PCA降维的聚类中心可视化', fontsize=15)
        plt.xlabel('主成分1', fontsize=12)
        plt.ylabel('主成分2', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_centers_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"聚类中心PCA可视化已保存至 {output_dir}/cluster_centers_pca.png")
    
except Exception as e:
    print(f"进行聚类分析时出错: {e}")
    import traceback
    traceback.print_exc()
    print("跳过聚类分析")

# 5. 生成用户画像总结
print("\n\n5. 用户画像总结")
print("-" * 50)

# 为了展示分组分布，我们需要计算全数据集的统计
print("计算全数据集的年龄分组分布...")
# 使用Dask创建分组，确保所有分区使用相同的 ordered 参数
# 显式指定 ordered=False 以避免不一致性
age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
age_labels = ['<18岁', '18-25岁', '26-35岁', '36-45岁', '46-55岁', '56-65岁', '>65岁']

ddf['age_group'] = ddf['age'].map_partitions(
    lambda x: pd.cut(x, bins=age_bins, labels=age_labels, ordered=False),
    meta=('age_group', 'category')
)

print("计算全数据集的收入分组分布...")
income_bins = [0, 100000, 300000, 500000, 700000, 1000000]
income_labels = ['低收入', '中低收入', '中等收入', '中高收入', '高收入']

ddf['income_group'] = ddf['income'].map_partitions(
    lambda x: pd.cut(x, bins=income_bins, labels=income_labels, ordered=False),
    meta=('income_group', 'category')
)

# 各维度分布统计，使用更安全的方式计算
print("计算各维度分布统计...")
try:
    # 更安全的计算分布的方法，避免 Categorical.ordered 错误
    age_counts = ddf['age_group'].value_counts().compute()
    age_dist = (age_counts / age_counts.sum() * 100).sort_index()
    
    gender_counts = ddf['gender'].value_counts().compute()
    gender_dist = (gender_counts / gender_counts.sum() * 100)
    
    income_counts = ddf['income_group'].value_counts().compute()
    income_dist = (income_counts / income_counts.sum() * 100).sort_index()
    
    country_counts = ddf['country'].value_counts().compute().head(5)
    country_dist = (country_counts / ddf.shape[0].compute() * 100)
    
    print("年龄分布:")
    for age_group, percentage in age_dist.items():
        print(f"  {age_group}: {percentage:.2f}%")
    
    print("\n性别分布:")
    for gender, percentage in gender_dist.items():
        print(f"  {gender}: {percentage:.2f}%")
    
    print("\n收入分布:")
    for income_group, percentage in income_dist.items():
        print(f"  {income_group}: {percentage:.2f}%")
    
    print("\n主要国家分布(前5名):")
    for country, percentage in country_dist.items():
        print(f"  {country}: {percentage:.2f}%")
    
except Exception as e:
    print(f"计算分布统计时出错: {e}")
    import traceback
    traceback.print_exc()
    print("跳过分布统计分析")

# 继续其他分析
if 'purchase_count' in ddf.columns:
    try:
        purchase_stats = ddf['purchase_count'].describe().compute()
        print(f"\n购买行为统计:")
        print(f"  平均购买次数: {purchase_stats['mean']:.2f}")
        print(f"  购买次数中位数: {purchase_stats['50%']:.0f}")
        print(f"  最大购买次数: {purchase_stats['max']:.0f}")
    except Exception as e:
        print(f"计算购买统计时出错: {e}")

print("\n活跃度:")
print(f"  活跃用户比例: {active_percent:.2f}%")
print(f"  非活跃用户比例: {inactive_percent:.2f}%")

# 用户注册年限分析
try:
    current_year = datetime.now().year
    ddf['account_age'] = current_year - ddf['registration_date'].dt.year
    account_age_stats = ddf['account_age'].describe().compute()
    
    print("\n用户账户年限:")
    print(f"  平均账户年限: {account_age_stats['mean']:.2f}年")
    print(f"  最长账户年限: {account_age_stats['max']:.0f}年")
except Exception as e:
    print(f"计算账户年限时出错: {e}")

# 6. 保存分析结果
print("\n\n6. 保存分析结果")
print("-" * 50)

# 将完整的处理结果保存为CSV
output_file = f'{output_dir}/analysis_results.csv'
print(f"将全部数据保存为CSV (用于报告生成)...")
# 保存全部数据
full_data = ddf.compute()
full_data.to_csv(output_file, index=False)
print(f"完整分析结果已保存至 {output_file}")

print("\n分析完成，详细结果请查看 analysis_output 目录下的图表和数据文件。")

end_time = datetime.now()
print(f"分析完成，用时: {end_time - start_time}")
