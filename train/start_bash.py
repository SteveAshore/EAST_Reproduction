import subprocess
import os

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.realpath(__file__))

# 设置 shell 脚本的路径
script_path = os.path.join(script_dir, 'run_east_dpo_test.sh')

# 使用 subprocess 启动 shell 脚本
try:
    subprocess.run(['bash', script_path], check=True)
    print("脚本成功启动！")
except subprocess.CalledProcessError as e:
    print(f"脚本执行失败: {e}")
except FileNotFoundError:
    print(f"无法找到脚本: {script_path}")
