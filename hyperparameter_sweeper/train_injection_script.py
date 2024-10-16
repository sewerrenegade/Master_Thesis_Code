if __name__ == "__main__":
    import sys
    import os
    import numpy

    # Append the parent directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
    from train import main
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