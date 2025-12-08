import os
import json
from collections import defaultdict
import argparse

def merge_results(result_dir, save_path):
    """
    Merge multi-process inference results and calculate average scores.
    
    Args:
        result_dir (str): Directory containing result files from each process, each file in JSON format.
        save_path (str): Path to save the merged statistical results.
    """
    # Initialize cumulative dictionaries
    dim_cnt_dict = defaultdict(int)
    dim_score_cn_dict = defaultdict(float)
    dim_score_en_dict = defaultdict(float)
    dim_score_strict_cn_dict = defaultdict(float)
    dim_score_strict_en_dict = defaultdict(float)
    
    # Read all process result files
    files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
    if not files:
        raise ValueError(f"No result files found in {result_dir}")
    print(f"Found {len(files)} result files, merging...")

    for fname in files:
        fpath = os.path.join(result_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            res = json.load(f)
        
        for dim, cnt in res.items():
            dim_cnt_dict[dim] += cnt['cnt']
            dim_score_cn_dict[dim] += cnt['cn_score']
            dim_score_en_dict[dim] += cnt['en_score']
            dim_score_strict_cn_dict[dim] += cnt['strict_cn_score']
            dim_score_strict_en_dict[dim] += cnt['strict_en_score']
    
    # Calculate average scores
    for dim in dim_cnt_dict.keys():
        dim_score_cn_dict[dim] /= dim_cnt_dict[dim]
        dim_score_en_dict[dim] /= dim_cnt_dict[dim]
        dim_score_strict_cn_dict[dim] /= dim_cnt_dict[dim]
        dim_score_strict_en_dict[dim] /= dim_cnt_dict[dim]

    # Dimension list
    dimension_list = [
        "Scene Understanding",
        "Image Theme",
        "Viewpoint of Capture",
        "Target Localization",
        "Spatial Relationship Understanding",
        "Object Counting",
        "Thermal Feature Understanding",
        "Action Recognition",
        "Thermal Feature Reasoning",
        "Commonsense Reasoning",
    ]

    # Print and save results
    score_dict = dict()
    print("=" * 100)
    for dim in dimension_list:
        if dim not in dim_cnt_dict:
            continue
        print(f"{dim:<35} cnt: {dim_cnt_dict[dim]:<5}  cn_score: {dim_score_cn_dict[dim]:.4f}  en_score: {dim_score_en_dict[dim]:.4f}")
        score_dict[dim] = {
            "cnt": dim_cnt_dict[dim],
            "cn_score": dim_score_cn_dict[dim],
            "en_score": dim_score_en_dict[dim],
            "avg_score": (dim_score_cn_dict[dim] + dim_score_en_dict[dim]) / 2,
            "strict_cn_score": dim_score_strict_cn_dict[dim],
            "strict_en_score": dim_score_strict_en_dict[dim],
            "strict_avg_score": (dim_score_strict_cn_dict[dim] + dim_score_strict_en_dict[dim]) / 2,
        }

    # Overall average
    score_dict["avg"] = {
        "cnt": sum(dim_cnt_dict.values()),
        "cn_score": sum(dim_score_cn_dict.values()) / len(dim_cnt_dict),
        "en_score": sum(dim_score_en_dict.values()) / len(dim_cnt_dict),
        "strict_cn_score": sum(dim_score_strict_cn_dict.values()) / len(dim_cnt_dict),
        "strict_en_score": sum(dim_score_strict_en_dict.values()) / len(dim_cnt_dict),
    }
    score_dict["avg"]["avg_score"] = (score_dict["avg"]["cn_score"] + score_dict["avg"]["en_score"]) / 2
    score_dict["avg"]["strict_avg_score"] = (score_dict["avg"]["strict_cn_score"] + score_dict["avg"]["strict_en_score"]) / 2

    assert score_dict["avg"]["cnt"] == 680, f"avg cnt is {score_dict['avg']['cnt']}"
    print(f"\n{'avg':<35} cnt: {score_dict['avg']['cnt']:<5}  cn_score: {score_dict['avg']['cn_score']:.4f}  en_score: {score_dict['avg']['en_score']:.4f}  avg_score: {score_dict['avg']['avg_score']:.4f}")
    print(f"avg \t cnt: {score_dict['avg']['cnt']} \t strict_cn_score: {score_dict['avg']['strict_cn_score']} \t strict_en_score: {score_dict['avg']['strict_en_score']} \t strict_avg_score: {score_dict['avg']['strict_avg_score']}")
    print("=" * 100)

    # Save as JSON file
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, indent=4, ensure_ascii=False)
    print(f"Merged results saved to {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge results from multiple processes.")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing result files.")
    parser.add_argument("--save_prefix", type=str, required=True, help="Prefix for saving merged results.")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    result_dir = os.path.join(args.result_dir, args.save_prefix, "outputs")
    if not os.path.exists(result_dir):
        raise ValueError(f"Result directory {result_dir} does not exist.")
    
    file_name = os.listdir(result_dir)[0].split(".")[0].split("_task")[0]
    if args.save_path is None:
        save_path = os.path.join(args.result_dir, args.save_prefix, f"merged_results/{file_name}.json")
    else:
        save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merge_results(result_dir, save_path)
