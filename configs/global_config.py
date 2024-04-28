class GlobalConfig:

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


    RESULTS_FOLDER_PATH = "results/"