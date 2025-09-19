import os

def count_json_files(directory):
    # 计数器初始化为0
    json_count = 0
    
    # 遍历指定目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 如果文件以.json结尾
            if file.endswith('.json'):
                json_count += 1
    
    return json_count

def main(directory):
    # 统计.json文件数量
    json_files = count_json_files(directory)
    
    # 输出统计结果
    print(f"该文件夹下共有 {json_files} 个 .json 文件")

if __name__ == "__main__":
    # 示例路径，可以替换成你需要的路径
    directory = "/home/yangmx/MRAG/Experiment/Generation/Output"  # 替换为实际路径
    main(directory)
