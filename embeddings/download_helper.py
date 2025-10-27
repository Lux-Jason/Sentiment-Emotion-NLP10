#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen模型下载帮助脚本 - 支持4B和8B模型下载
使用镜像加速下载，支持根据设备情况选择合适的模型
"""
import os
import sys
import argparse

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_model(model_size, use_mirror=True):
    """
    下载指定大小的Qwen模型
    
    Args:
        model_size: 模型大小，'4B' 或 '8B'
        use_mirror: 是否使用镜像加速
    """
    model_configs = {
        '4B': {
            'name': 'Qwen/Qwen3-Embedding-4B',
            'size': '~8GB',
            'memory': '8-10GB',
            'description': '中等大小，性能优秀，适合大多数用户'
        },
        '8B': {
            'name': 'Qwen/Qwen3-Embedding-8B', 
            'size': '~16GB',
            'memory': '16GB+',
            'description': '最大模型，MTEB排行榜第一，需要更多资源'
        }
    }
    
    if model_size not in model_configs:
        print(f"❌ 错误: 不支持的模型大小 '{model_size}'，请选择 '4B' 或 '8B'")
        return False
    
    config = model_configs[model_size]
    model_name = config['name']
    
    print("=" * 80)
    print("Qwen模型下载工具")
    print("=" * 80)
    print(f"模型大小: {model_size}")
    print(f"模型名称: {model_name}")
    print(f"模型大小: {config['size']}")
    print(f"内存需求: {config['memory']}")
    print(f"描述: {config['description']}")
    
    if use_mirror:
        print("使用镜像: https://hf-mirror.com")
    else:
        print("使用官方源: HuggingFace")
    print("=" * 80)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"\n🚀 开始下载模型: {model_name}")
        print("⏳ 请耐心等待，首次下载需要较长时间...")
        print(f"   4B模型约8GB，8B模型约16GB")
        print()
        
        # 设置缓存目录
        cache_folder = f'./model_cache/Qwen3-Embedding-{model_size}'
        
        # 下载模型
        model = SentenceTransformer(
            model_name,
            device='cpu',
            trust_remote_code=True,
            cache_folder=cache_folder
        )
        
        print(f"\n✅ 模型下载成功！")
        print(f"📁 缓存位置: {cache_folder}")
        print(f"🎯 模型已准备就绪，可以开始使用！")
        
        return True
        
    except ImportError:
        print("❌ 错误: 请先安装依赖")
        print("运行: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 尝试使用魔法上网")
        print("3. 手动从 https://hf-mirror.com/Qwen/Qwen3-Embedding-4B 下载")
        print("4. 手动从 https://hf-mirror.com/Qwen/Qwen3-Embedding-8B 下载")
        return False

def print_model_comparison():
    """打印模型对比信息"""
    print("=" * 80)
    print("Qwen模型对比")
    print("=" * 80)
    print(f"{'指标':<20} {'4B':<25} {'8B':<25}")
    print("-" * 80)
    print(f"{'参数量':<20} {'4B':<25} {'8B':<25}")
    print(f"{'模型大小':<20} {'~8GB':<25} {'~16GB':<25}")
    print(f"{'内存需求':<20} {'8-10GB':<25} {'16GB+':<25}")
    print(f"{'C-MTEB中文':<20} {'72.27':<25} {'73.84':<25}")
    print(f"{'MTEB多语言':<20} {'69.45':<25} {'70.58':<25}")
    print(f"{'相对速度':<20} {'1.4x (快40%)':<25} {'1.0x (基准)':<25}")
    print(f"{'下载时间':<20} {'~12分钟':<25} {'~25分钟':<25}")
    print("-" * 80)
    print(f"{'推荐场景':<20}")
    print(f"  4B: 大多数用户，平衡性能和资源使用")
    print(f"  8B: 追求最佳性能且有足够GPU内存的用户")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Qwen模型下载工具')
    parser.add_argument('--model', '-m', choices=['4B', '8B', 'both'], 
                       default='both', help='选择要下载的模型大小 (4B, 8B, both)')
    parser.add_argument('--no-mirror', action='store_true', 
                       help='不使用镜像，直接从官方源下载')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='显示模型对比信息')
    
    args = parser.parse_args()
    
    if args.compare:
        print_model_comparison()
        return
    
    use_mirror = not args.no_mirror
    
    if args.model == 'both':
        print("📦 将下载4B和8B两个模型")
        print("💡 提示: 您可以根据设备情况后续选择使用哪个模型\n")
        
        success_4b = download_model('4B', use_mirror)
        print("\n" + "="*50 + "\n")
        success_8b = download_model('8B', use_mirror)
        
        if success_4b and success_8b:
            print("\n🎉 所有模型下载完成！")
            print("💡 使用建议:")
            print("   - 如果GPU内存 < 12GB，建议使用4B模型")
            print("   - 如果GPU内存 >= 16GB，可以使用8B模型获得最佳性能")
        else:
            print("\n⚠️ 部分模型下载失败，请检查错误信息")
    else:
        download_model(args.model, use_mirror)

if __name__ == "__main__":
    main()
