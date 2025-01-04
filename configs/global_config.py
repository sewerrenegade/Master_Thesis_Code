


def block_useless_warnings():
    import warnings
    warnings.filterwarnings("ignore", message="xFormers is available")

def disable_plt_interactive_gui():
    import matplotlib
    matplotlib.use('Agg')
    
class GlobalConfig:
    block_useless_warnings()
    #disable_plt_interactive_gui()
    HTMLStart = "<HTML><HEAD><TITLE>{0}</TITLE></HEAD><BODY style=\"background-color:#888888\"><CENTER>{1}<BR><IMG SRC=\"../extras/images/large-crisis-blackadder.jpg\" ALIGN=\"BOTTOM\"> </CENTER><HR>"
    HTMLEnd = "<HR></BODY></HTML>"
    ColoredLog = "<p style=\"color:{0}\">"
    HTML_COLORS = [
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#0000FF",  # Blue
    "#FFFF00",  # Yellow
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cyan
    "#800080",  # Purple
    "#FFFFFF",  # White
    "#000000"   # Black
    ]
    DEV_MODE = True
    PATH_TO_SAVED_MODELS = "trained_models/"
    PATH_TO_SERIALIZED_SESSION_NAMES = "extras/session_names.pkl"
    CUSTOM_METRICS = []

    EMBEDDING_DATA_FOLDER_PATH = "data/EMBEDDING/v1/"
    RESULTS_DATA_FOLDER_PATH = "results/data/v2/"
    RESULTS_FOLDER_PATH = "results/"
    CONNECTIVITY_DP_VID_PATH = "connectivity_dp_viz/"
    CONNECTIVITY_DP_SWEEPER_PATH = RESULTS_FOLDER_PATH + "dp_sweep_results/"
    MNIST_INTER_CLASS_DIST = "MNIST_interinstance_distances/"
    FashionMNIST_INTER_CLASS_DIST = "FashionMNIST_interinstance_distances/"
    EMBEDDING_RESULTS = "embedding_results"
    NAME_OF_LABELED_EMBEDDED_FEATURES = "labeled_embedded_features"
    NAME_OF_STATS_OF_EMBEDDED_FEATURES = "stats_of_embedded_features" 

    DOWNPROJECTION_TEST_DIMENSIONS = [1,2,3,4,6,8,12,16,32]
    EMBEDDING_METHODS = ["PHATE","ISOMAP","PCA","TSNE","UMAP"]
    
    TEMP_RESULTS_FOLDER = "results/temp_results/"
    
    LIST_OF_ALL_DATASETS = [
        "MNSIT",
        "FMNIST",
        "CIFAR10",
        "SCEMILA/image_data",
        "SCEMILA/fnl34_feature_data",
        
    ]