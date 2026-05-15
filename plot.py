import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# Aesthetics configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['axes.linewidth'] = 1.0

def get_last_round_metrics(csv_path):
    """Lấy kết quả của round cuối cùng trong mỗi task."""
    df = pd.read_csv(csv_path)
    if 'task' not in df.columns:
        return pd.DataFrame()
    last_rounds = df.groupby('task').last().reset_index()
    return last_rounds

def plot_framework(dataset_name, metric='f1_mac', y_label='macro-F1 (%)', base_dir=r'D:\IDPS\HFIN\IDPS\logs'):
    """Vẽ biểu đồ so sánh các phương pháp giống như ảnh tham khảo."""
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return
        
    methods = ['replay', 'icarl', 'wa', 'der']
    
    # Map các thư mục variant thành nhãn, kiểu đường (linestyle), marker và độ đậm nhạt của màu
    # Theo như ảnh: HFL (nhạt nhất), beta=0 (gạch gạch nhạt), beta=0.5 (đậm vừa), beta=1 (đậm nhất chấm chấm)
    # Lưu ý: Các thư mục bạn cần có tương ứng là WTO_NONE_OLD, WTO_NONE, WTO_0.5, WTO_1.0
    variants = [
        ('WTO_NONE', 'HFL', '-', 'v', 0.4),                 # Solid line, down-triangle, light color
        ('WTO', 'HFL+WTO^{\\beta=0.5}', '-', 'D', 0.8),     # Solid line, diamond, medium-dark color
    ]
    
    colors = {
        'replay': plt.cm.Blues,
        'icarl': plt.cm.Oranges,
        'wa': plt.cm.Greens,
        'der': plt.cm.Purples
    }
    
    method_names = {
        'replay': 'Replay',
        'icarl': 'iCaRL',
        'wa': 'WA',
        'der': 'DER'
    }
    
    # Khởi tạo 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
    
    handles_dict = {}
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        method_dir = os.path.join(dataset_dir, method)
        
        # Nếu thư mục method chưa có, ta cứ để đồ thị trống nhưng vẫn có title/axes
        color_map = colors[method]
        
        # Xác định xticks mặc định theo dataset
        ds_lower = dataset_name.lower()
        if 'uq-nids' in ds_lower or 'uq_nids' in ds_lower:
            xticks = [5, 9, 13, 17, 21]
        elif 'ton-iot' in ds_lower or 'ton_iot' in ds_lower:
            xticks = [2, 4, 6, 8, 10]
        else:
            xticks = []

        if os.path.exists(method_dir):
            for variant_folder, label_suffix, linestyle, marker, color_intensity in variants:
                csv_file = os.path.join(method_dir, variant_folder, f'metrics_{method}.csv')
                
                if not os.path.exists(csv_file):
                    continue
                    
                df = get_last_round_metrics(csv_file)
                if df.empty:
                    continue
                
                # Tính số lượng class trên x-axis
                if 'uq-nids' in ds_lower or 'uq_nids' in ds_lower:
                    x_vals = 5 + df['task'] * 4
                elif 'ton-iot' in ds_lower or 'ton_iot' in ds_lower:
                    x_vals = 2 + df['task'] * 2
                else:
                    x_vals = df['task']
                    if not xticks: xticks = df['task'].unique()
                    
                y_vals = df[metric]
                
                label = f'{method_names[method]}+{label_suffix}'
                color = color_map(color_intensity)
                
                line, = ax.plot(x_vals, y_vals, 
                                label=label, 
                                color=color, 
                                linestyle=linestyle, 
                                marker=marker, 
                                markersize=6, 
                                linewidth=1.5)
                
                # Lưu line để làm legend chung
                handles_dict[label] = line
                
        ax.set_xlabel('Number of Classes')
        ax.set_ylabel(y_label)
        if len(xticks) > 0:
            ax.set_xticks(xticks)
            
    # Lấy handles và labels để tạo Legend chung ở trên cùng
    ordered_handles = []
    ordered_labels = []
    for label, handle in handles_dict.items():
        ordered_labels.append(label)
        ordered_handles.append(handle)
        
    if ordered_handles:
        fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=8, frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # Chừa khoảng trống cho legend
    
    # Save biểu đồ
    save_path = os.path.join(base_dir, f'{dataset_name}_{metric}_plot.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Saved plot to: {save_path}')

if __name__ == "__main__":
    datasets = ['NF-ToN-IoT-v2', 'NF-UQ-NIDS-v2']
    for ds in datasets:
        print(f"Plotting for {ds}...")
        plot_framework(ds, metric='f1_mac', y_label='macro-F1 (%)')
        plot_framework(ds, metric='acc', y_label='Accuracy (%)')
