import os
import pandas as pd
import re
from collections import defaultdict


def find_fold_csv_files(directory="."):
    """
    查找目录中所有fold*_train.csv, fold*_test.csv, fold*_val.csv文件，并按fold分组
    """
    fold_pattern = re.compile(r"fold(\d+)_(train|test|val)\.csv")
    fold_files = defaultdict(dict)

    for filename in os.listdir(directory):
        match = fold_pattern.match(filename)
        if match:
            fold_num = match.group(1)
            file_type = match.group(2)
            fold_files[fold_num][file_type] = os.path.join(directory, filename)

    return fold_files


def check_reaction_duplicates(fold_files):
    """
    检查每个fold组中train.csv, test.csv, val.csv之间是否有相同的反应
    """
    results = {}

    for fold_num, files in fold_files.items():
        if len(files) < 2:
            print(f"Skip fold {fold_num}: 不完整的fold组（缺少文件）")
            continue

        # 读取文件
        dfs = {}
        for file_type, filepath in files.items():
            try:
                dfs[file_type] = pd.read_csv(filepath)
            except Exception as e:
                print(f"无法读取 {filepath}: {e}")
                continue

        # 检查每对文件之间的重复
        fold_results = {}
        file_pairs = [(a, b) for i, a in enumerate(dfs.keys()) for b in list(dfs.keys())[i + 1:]]

        for file_a, file_b in file_pairs:
            if 'reaction' not in dfs[file_a].columns or 'reaction' not in dfs[file_b].columns:
                fold_results[f"{file_a}-{file_b}"] = "缺少'reaction'列"
                continue

            reactions_a = set(dfs[file_a]['reaction'])
            reactions_b = set(dfs[file_b]['reaction'])
            duplicates = reactions_a.intersection(reactions_b)

            fold_results[f"{file_a}-{file_b}"] = {
                "重复数量": len(duplicates),
                "重复比例_A": len(duplicates) / len(reactions_a) if len(reactions_a) > 0 else 0,
                "重复比例_B": len(duplicates) / len(reactions_b) if len(reactions_b) > 0 else 0,
                "样本量_A": len(reactions_a),
                "样本量_B": len(reactions_b),
                "重复例子": list(duplicates)[:5] if duplicates else []
            }

        results[fold_num] = fold_results

    return results


def print_results(results):
    """打印结果的简洁摘要"""
    print("\n=== 反应重复检查结果 ===")

    for fold_num, fold_results in results.items():
        print(f"\nFold {fold_num}:")

        for pair, pair_results in fold_results.items():
            if isinstance(pair_results, str):
                print(f"  {pair}: {pair_results}")
                continue

            print(f"  {pair}:")
            print(f"    重复数量: {pair_results['重复数量']}")
            print(f"    样本量: {pair_results['样本量_A']} / {pair_results['样本量_B']}")
            print(f"    重复比例: {pair_results['重复比例_A']:.2%} / {pair_results['重复比例_B']:.2%}")

            if pair_results['重复例子']:
                print(f"    重复例子 (前5个):")
                for i, example in enumerate(pair_results['重复例子'], 1):
                    # 显示缩略版本的反应式
                    short_example = example[:50] + "..." if len(example) > 50 else example
                    print(f"      {i}. {short_example}")


def main():
    # 查找所有fold组文件
    fold_files = find_fold_csv_files()
    if not fold_files:
        print("未找到任何fold组文件")
        return

    # 检查重复
    results = check_reaction_duplicates(fold_files)

    # 打印结果
    print_results(results)

    # 总结
    print("\n=== 总结 ===")
    for fold_num, fold_results in results.items():
        has_duplicates = any(
            isinstance(r, dict) and r['重复数量'] > 0
            for r in fold_results.values()
        )
        status = "❌ 存在重复" if has_duplicates else "✅ 无重复"
        print(f"Fold {fold_num}: {status}")


if __name__ == "__main__":
    main()