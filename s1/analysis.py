#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA StereoBM Performance Analysis Tool
用于分析敏感性实验的CSV数据并生成统计报告
"""

import pandas as pd
import sys
import os
from pathlib import Path

def analyze_sensitivity_results(csv_file, experiment_type):
    """分析敏感性实验结果"""
    
    print(f"\n{'='*70}")
    print(f"Analysis Report: {experiment_type}")
    print(f"CSV File: {csv_file}")
    print(f"{'='*70}\n")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    # 根据实验类型进行分析
    if 'max_disp' in experiment_type.lower():
        return analyze_max_disp(df)
    elif 'window_size' in experiment_type.lower():
        return analyze_window_size(df)
    elif 'block_size' in experiment_type.lower():
        return analyze_block_size(df)
    elif 'multi_dataset' in experiment_type.lower():
        return analyze_multi_dataset(df)
    else:
        return analyze_generic(df)

def analyze_max_disp(df):
    """Max_disp参数敏感性分析"""
    
    # 转换时间列为数值
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df.dropna(subset=['time_ms'])
    
    # 分组统计
    summary = df.groupby(['max_disp', 'version'])['time_ms'].agg([
        ('avg_time', 'mean'),
        ('std_dev', 'std'),
        ('min_time', 'min'),
        ('max_time', 'max')
    ]).reset_index()
    
    print("Average Time (ms) by max_disp and Version:")
    print(summary.to_string(index=False))
    print()
    
    # 计算加速比
    print("\nSpeedup Ratios (V1 as baseline):")
    for max_disp in sorted(df['max_disp'].unique()):
        subset = summary[summary['max_disp'] == max_disp]
        v1_time = subset[subset['version'] == 'v1']['avg_time'].values[0]
        
        print(f"\nmax_disp = {max_disp}:")
        for _, row in subset.iterrows():
            speedup = v1_time / row['avg_time']
            print(f"  {row['version'].upper():3s}: {row['avg_time']:8.3f} ms  (Speedup: {speedup:6.2f}x)")
    
    return summary

def analyze_window_size(df):
    """Window_size参数敏感性分析"""
    
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df.dropna(subset=['time_ms'])
    
    summary = df.groupby(['window_size', 'version'])['time_ms'].agg([
        ('avg_time', 'mean'),
        ('std_dev', 'std')
    ]).reset_index()
    
    print("Average Time (ms) by window_size and Version:")
    print(summary.to_string(index=False))
    print()
    
    # 计算增长率
    print("\nTime Growth Analysis:")
    for version in sorted(df['version'].unique()):
        subset = summary[summary['version'] == version].sort_values('window_size')
        print(f"\n{version.upper()}:")
        
        first_time = None
        for _, row in subset.iterrows():
            if first_time is None:
                first_time = row['avg_time']
                growth = 1.0
            else:
                growth = row['avg_time'] / first_time
            
            window_area = (2 * (row['window_size']//2) + 1) ** 2
            print(f"  window_size={row['window_size']:2d} (area={window_area:3d}): {row['avg_time']:8.3f} ms (x{growth:.2f})")
    
    return summary

def analyze_block_size(df):
    """Block尺寸敏感性分析"""
    
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df.dropna(subset=['time_ms'])
    
    summary = df.groupby('block_size')['time_ms'].agg([
        ('avg_time', 'mean'),
        ('std_dev', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    summary = summary.sort_values('avg_time')
    
    print("Block Size Performance (CUDA V5):")
    print(summary.to_string(index=False))
    print()
    
    # 找最优block
    best_block = summary.iloc[0]
    best_time = best_block['avg_time']
    
    print(f"\nOptimal Configuration:")
    print(f"  Block Size: {best_block['block_size']}")
    print(f"  Average Time: {best_time:.3f} ms")
    print(f"\nPerformance Delta (vs best):")
    
    for _, row in summary.iterrows():
        delta = row['avg_time'] - best_time
        pct_delta = (delta / best_time) * 100
        print(f"  {row['block_size']:6s}: +{delta:6.3f} ms ({pct_delta:+6.2f}%)")
    
    return summary

def analyze_multi_dataset(df):
    """多数据集综合分析"""
    
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df.dropna(subset=['time_ms'])
    
    summary = df.groupby(['dataset', 'image_size', 'version'])['time_ms'].agg([
        ('avg_time', 'mean'),
        ('std_dev', 'std')
    ]).reset_index()
    
    print("Performance Summary Across Datasets:")
    print(summary.to_string(index=False))
    print()
    
    # 计算加速比和统计
    print("\nSpeedup Analysis (V1 as baseline):")
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_summary = summary[summary['dataset'] == dataset]
        v1_subset = dataset_summary[dataset_summary['version'] == 'v1']
        
        if len(v1_subset) == 0:
            continue
        
        v1_time = v1_subset['avg_time'].values[0]
        img_size = dataset_summary['image_size'].values[0]
        
        print(f"\n{dataset:10s} ({img_size}):")
        
        for _, row in dataset_summary.iterrows():
            speedup = v1_time / row['avg_time']
            print(f"  {row['version'].upper():3s}: {row['avg_time']:8.3f} ms  (Speedup: {speedup:6.2f}x)")
    
    return summary

def analyze_generic(df):
    """通用分析"""
    print("Generic Analysis:")
    print(df.describe())
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analysis.py <csv_file> [experiment_type]")
        print("\nExample:")
        print("  python3 analysis.py results/sensitivity_max_disp_20260602_120000/summary.csv")
        print("  python3 analysis.py results/sensitivity_window_size_20260602_120000/summary.csv")
        return
    
    csv_file = sys.argv[1]
    experiment_type = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.dirname(csv_file))
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        return
    
    analyze_sensitivity_results(csv_file, experiment_type)

if __name__ == '__main__':
    main()
