import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from train import main


if __name__ == "__main__":
    sys.argv = [arg.lstrip('--') for arg in sys.argv]
    main()
#   #setup_and_start_training(5)