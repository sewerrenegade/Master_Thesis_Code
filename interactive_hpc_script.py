#!/bin/env python
print("Strating Interactive script")

#salloc --nodes=1 --cpus-per-task=4 --mem=20G --time=3:00:00 --partition=interactive_gpu_p --qos=interactive_gpu --nice=10000 --gres=gpu:1 --job-name=std_master_allocation
#u need to make this file executable
#insert main function here


if __name__ == '__main__':
    from train import main
    main()
