import pandas as pd
import os
import glob

def analyze_logs():
    base_dir = r'D:\IDPS\HFIN\IDPS\logs'
    datasets = ['NF-ToN-IoT-v2', 'NF-UQ-NIDS-v2']
    methods = ['der', 'wa', 'icarl']
    variants = ['WTO', 'WTO_NONE']
    
    results = []
    
    for ds in datasets:
        for method in methods:
            for var in variants:
                csv_path = os.path.join(base_dir, ds, method, var, f'metrics_{method}.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        last = df.iloc[-1]
                        results.append({
                            'Dataset': ds,
                            'Method': method.upper(),
                            'WTO': 'ON' if var == 'WTO' else 'OFF',
                            'Tasks': int(last['task']) + 1,
                            'Final Acc': last['acc'],
                            'Final F1': last['f1_mac'],
                            'Final Loss': last['loss']
                        })
    
    if results:
        res_df = pd.DataFrame(results)
        print("\n=== SUMMARY PERFORMANCE COMPARISON ===")
        print(res_df.to_string(index=False))
        
        # Calculate WTO Improvement
        print("\n=== WTO IMPROVEMENT ANALYSIS ===")
        for ds in datasets:
            for method in methods:
                on = res_df[(res_df['Dataset'] == ds) & (res_df['Method'] == method.upper()) & (res_df['WTO'] == 'ON')]
                off = res_df[(res_df['Dataset'] == ds) & (res_df['Method'] == method.upper()) & (res_df['WTO'] == 'OFF')]
                
                if not on.empty and not off.empty:
                    acc_diff = on.iloc[0]['Final Acc'] - off.iloc[0]['Final Acc']
                    f1_diff = on.iloc[0]['Final F1'] - off.iloc[0]['Final F1']
                    print(f"[{ds} | {method.upper()}]: Acc Gain: {acc_diff:+.2f}%, F1 Gain: {f1_diff:+.2f}%")
    else:
        print("No results found in logs.")

if __name__ == '__main__':
    analyze_logs()
