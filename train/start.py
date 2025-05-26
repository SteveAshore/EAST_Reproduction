import subprocess
import os
import sys
from pathlib import Path

def run_shell_script():
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    
    # 构造脚本的绝对路径
    script_path = current_dir / "run_east_dpo_test.sh"
    
    # 检查脚本是否存在
    if not script_path.exists():
        print(f"错误：脚本文件 {script_path} 不存在")
        sys.exit(1)
    
    # 检查执行权限（仅限 Unix 系统）
    if not os.access(script_path, os.X_OK):
        print(f"正在为脚本添加执行权限...")
        script_path.chmod(0o755)  # 添加 rwxr-xr-x 权限
    
    try:
        # 执行脚本并捕获输出
        result = subprocess.run(
            [str(script_path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        print("执行成功，输出内容：\n" + result.stdout)
    
    except subprocess.CalledProcessError as e:
        print(f"执行失败，错误信息：\n{e.output}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    run_shell_script()
