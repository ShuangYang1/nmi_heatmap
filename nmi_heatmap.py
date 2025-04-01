from sklearn.metrics.cluster import normalized_mutual_info_score
from dtaidistance import dtw
import numpy as np

def extract(filename):
    ram = []
    cpu = []
    pfs_recv = []
    pfs_send = []
    power = []
    with open(filename, "r") as infile:
        for line in infile.readlines():
            if line[:3] == "ram":
                tot = line.split(" ")[2]
                free = line.split(" ")[4]
                used = (
                    (int(tot) - int(free)) * 4 * 1024 / 1024 / 1024 / 1024 / 512 * 100
                )
                ram.append(used)
            if line[:3] == "pfs":
                recv = int(line.split(" ")[3]) / 1024 / 1024 / 1024 / 100 * 100
                send = int(line.split(" ")[5]) / 1024 / 1024 / 1024 / 100 * 100
                pfs_recv.append(recv)
                pfs_send.append(send)
            if line[:7] == "cpu_tot":
                used = float(line.split(" ")[2])
                sys = float(line.split(" ")[4])
                iowait = float(line.split(" ")[6])
                cpu.append(used+sys+iowait)
            if line[:5] == "Power":
                p = int(line.split()[1]) / 500 * 100
                power.append(p)
    return ram, cpu, pfs_recv, pfs_send, power

def nmi(data1, data2):
    """计算归一化互信息，自动将连续值离散化"""
    # 将连续值离散化为10个区间
    bins = np.linspace(min(min(data1), min(data2)), 
                      max(max(data1), max(data2)), 10)
    digitized1 = np.digitize(data1, bins)
    digitized2 = np.digitize(data2, bins)
    
    # 计算NMI
    nmi_score = normalized_mutual_info_score(digitized1, digitized2, average_method='geometric')
    return nmi_score

def dtw_nmi(data1, data2):
    """
    使用DTW对齐后计算NMI
    参数:
        data1: 第一个时间序列
        data2: 第二个时间序列
    返回:
        对齐后的NMI值
    """
    # 转换为numpy数组并确保是1D数组
    s1 = np.array(data1).flatten()
    s2 = np.array(data2).flatten()
    
    # 计算DTW对齐路径
    path = dtw.warping_path(s1, s2)
    
    # 按照对齐路径重新采样
    aligned_s1 = np.array([s1[i] for i, _ in path])
    aligned_s2 = np.array([s2[j] for _, j in path])
    
    # 计算NMI
    return nmi(aligned_s1, aligned_s2)


def calculate_pairwise_nmi(data_dict):
    """
    计算字典中所有键值对之间的两两互信息
    参数:
        data_dict: 包含不同软件数据的字典
    返回:
        包含两两互信息得分的字典
    """
    software_names = list(data_dict.keys())
    nmi_results = {}
    
    for i in range(len(software_names)):
        for j in range(i+1, len(software_names)):
            name1 = software_names[i]
            name2 = software_names[j]
            score = dtw_nmi(data_dict[name1], data_dict[name2])
            nmi_results[f"{name1}-{name2}"] = score
            
    return nmi_results

def plot_heatmap(heatmap_data, softwares, output_path):
    """
    绘制并保存热图
    参数:
        heatmap_data: 热图数据矩阵
        softwares: 软件名称列表
        output_path: 输出图片路径
    """
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形并设置高DPI
    plt.figure(figsize=(12,10), dpi=300)
    ax = plt.gca()
    
    # 绘制热图
    im = plt.imshow(heatmap_data, cmap='bwr', vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('NMI Score', fontsize=12)
    
    # 设置刻度标签
    plt.xticks(range(len(softwares)), softwares, rotation=45, fontsize=12)
    plt.yticks(range(len(softwares)), softwares, fontsize=12)
    
    # 设置标题
    plt.title('各软件资源需求特征互信息热图', fontsize=14, pad=20)
    
    # 添加数值标签
    for i in range(len(softwares)):
        for j in range(len(softwares)):
            if i != j:
                ax.text(j, i, f"{heatmap_data[i,j]:.2f}", 
                       ha="center", va="center", 
                       color="w", fontsize=12)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热图已保存为: {output_path}")

def calculate_and_plot_nmi_heatmap(softwares, data_dict, output_path):
    """
    计算互信息并绘制热图
    参数:
        softwares: 软件名称列表
        data_dict: 包含各指标数据的字典
        output_path: 输出图片路径
    返回:
        metrics: 各指标互信息结果
        combined_nmi: 综合互信息结果
    """
    # 计算各指标的两两互信息
    metrics = {
        'cpu': calculate_pairwise_nmi(data_dict['cpu']),
        'ram': calculate_pairwise_nmi(data_dict['ram']),
        'pfs_recv': calculate_pairwise_nmi(data_dict['pfs_recv']),
        'pfs_send': calculate_pairwise_nmi(data_dict['pfs_send']),
        'power': calculate_pairwise_nmi(data_dict['power'])
    }
    
    # 合并所有指标的互信息
    combined_nmi = {}
    for pair in metrics['cpu'].keys():
        scores = [m[pair] for m in metrics.values()]
        combined_nmi[pair] = np.mean(scores)
    
    # 生成热图数据
    heatmap_data = np.zeros((len(softwares), len(softwares)))
    for i in range(len(softwares)):
        for j in range(i+1, len(softwares)):
            pair = f"{softwares[i]}-{softwares[j]}"
            heatmap_data[i,j] = combined_nmi[pair]
            heatmap_data[j,i] = combined_nmi[pair]  # 对称矩阵
    
    # 绘制热图
    plot_heatmap(heatmap_data, softwares, output_path)
    
    return metrics, combined_nmi

if __name__ == "__main__":
    softwares=['BWA','Bismark','Cellranger','GATK','SPAdes','STAR']
    all_cpu,all_ram,all_pfs_recv,all_pfs_send,all_power={},{},{},{},{}
    for software in softwares:
        ram, cpu, pfs_recv, pfs_send, power = extract(f'mon/{software.lower()}.txt')
        all_cpu[software]=cpu
        all_ram[software]=ram
        all_pfs_recv[software]=pfs_recv
        all_pfs_send[software]=pfs_send
        all_power[software]=power
    
    # 准备数据字典
    data_dict = {
        'cpu': all_cpu,
        'ram': all_ram,
        'pfs_recv': all_pfs_recv,
        'pfs_send': all_pfs_send,
        'power': all_power
    }
    print(data_dict)
    # 调用函数计算并绘制热图
    output_path = './software_nmi_heatmap.png'
    metrics, combined_nmi = calculate_and_plot_nmi_heatmap(softwares, data_dict, output_path)
    
    # 打印结果
    print("各指标互信息结果:")
    for metric, nmi_dict in metrics.items():
        print(f"\n{metric.upper()}互信息:")
        for pair, score in nmi_dict.items():
            print(f"{pair}: {score:.4f}")
    
    print("\n综合互信息结果:")
    for pair, score in combined_nmi.items():
        print(f"{pair}: {score:.4f}")