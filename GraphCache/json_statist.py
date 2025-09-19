import os

original_graph_dir = '/home/ningxy/Experiment/GraphCache/TAT-DQA/OriginalGraph'
final_graph_dir = '/home/ningxy/Experiment/GraphCache/TAT-DQA/FinalGraph'

def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

original_count = count_files(original_graph_dir)
final_count = count_files(final_graph_dir)

print(f'OriginalGraph 文件数: {original_count}')
print(f'FinalGraph 文件数: {final_count}')