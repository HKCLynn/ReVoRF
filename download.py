import os
import subprocess
import requests
from git import Repo

# 步骤1：克隆GitHub库
repo_url = "https://github.com/isl-org/DPT"
repo_dir = "DPT_model"
if not os.path.exists(repo_dir):
    print(f"Cloning the repository from {repo_url}...")
    Repo.clone_from(repo_url, repo_dir)
    print("Repository cloned successfully!")
else:
    print(f"Repository already exists at {repo_dir}.")

# 步骤2：下载模型文件
model_url = "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
model_path = os.path.join(repo_dir+'/weights/', "dpt_hybrid-midas-501f0c75.pt")
print(model_path)
if not os.path.exists(model_path):
    print(f"Downloading the model from {model_url}...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully!")
else:
    print(f"Model file already exists at {model_path}.")



repo_dir = "DPT_model"
os.chdir(repo_dir)
datasets=os.listdir('../data')
if not len(datasets):
    print("There are not any datasets")
else:
    if "nerf_synthetic" in datasets:
        for scene in os.listdir('../data/nerf_synthetic'):
            input_path='../data/nerf_synthetic/train/'+scene
            output_path='../depths/nerf_synthetic/'+scene
            print(input_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            subprocess.run(["python", "run_monodepth.py", "--input_path", input_path, "--output_path", output_path])
    if "nerf_llff_data" in datasets:
        for scene in os.listdir('../data/nerf_llff_data'):
            input_path='../data/nerf_llff_data/images_8/'+scene
            output_path='../depths/nerf_llff_data/'+scene
            print(input_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            subprocess.run(["python", "run_monodepth.py", "--input_path", input_path, "--output_path", output_path])









