import sys
import os

# Get the parent directory (one level up)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
from train import main


if __name__ == "__main__":
    base_config_path = None
    for item in sys.argv:
        if item.startswith('--base_config_path='):
            path = item.split('=', 1)[1]
            #sys.argv.remove(item)
            print(f" dis is za way {path}")
            break
    print(sys.argv)
    last_slash_index = path.rfind('/')
    if last_slash_index != -1:
        folder_path = path[:last_slash_index + 1]
        file_name = path[last_slash_index + 1:]
    else:
        folder_path = ''
        file_name = path
    main(config_path = folder_path, config_name= file_name)