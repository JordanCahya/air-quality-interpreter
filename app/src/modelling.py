# Import necessary library
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from datetime import datetime
from tqdm import tqdm
import yaml
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import hashlib

def load_parameter(parameter_direction):
    with open(parameter_direction,'r') as file:
        params = yaml.safe_load(file)
    
    return params

# Load params
params = load_parameter("app/config/configuration_file_1.yaml")

# Show params
params

# Load variables
X_rus = joblib.load("app/processed/X_rus.pkl")
y_rus = joblib.load("app/processed/y_rus.pkl")

X_ros = joblib.load("app/processed/X_ros.pkl")
y_ros = joblib.load("app/processed/y_ros.pkl")

X_sm = joblib.load("app/processed/X_sm.pkl")
y_sm = joblib.load("app/processed/y_sm.pkl")

X_train = joblib.load("app/processed/X_train_feng.pkl")
y_train = joblib.load("app/processed/y_train_feng.pkl")
 
X_valid = joblib.load("app/processed/X_valid_feng.pkl")
y_valid = joblib.load("app/processed/y_valid_feng.pkl")

X_test = joblib.load("app/processed/X_test_feng.pkl")
y_test = joblib.load("app/processed/y_test_feng.pkl")

# Create a baseline model
y_train.value_counts(normalize = True)

# Sanity check
print(X_rus.shape, X_ros.shape, X_sm.shape, X_valid.shape, X_test.shape)

# Create time stamp for current date and time
def time_stamp():
    return datetime.now()

# Create log template
def create_log_template():
    
    """
    This function is to create a log template to store the training log.
    
    :return logger: <dictionary> an empty log template
    """
    
    logger = {
        "model_name" : [],
        "model_uid" : [],
        "training_time" : [],
        "training_date" : [],
        "performance" : [],
        "f1_score_avg" : [],
        "data_configurations" : [],
    }

    return logger

def training_log_updater(current_log, log_path):
    current_log = current_log.copy()
    
    """
    This function updates the training log file with the current log data.

    :param current_log: <dictionary> dictionary containing the current training log data
    :param log_path: <str> path to the training log file
    :return last_log: <list> updated training log
    """

    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    except FileNotFoundError as ffe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    last_log.append(current_log)

    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    return last_log

# Create model object
lgr_baseline = LogisticRegression()
dct_baseline = DecisionTreeClassifier()
rfc_baseline = RandomForestClassifier()
knn_baseline = KNeighborsClassifier()
xgb_baseline = XGBClassifier()

# Create dictionary
list_of_model = {
    "normal" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""}
        ],
    "undersampling" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""}
        ],
    "oversampling" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""}
        ],
    "smote" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""}
        ],
    }

def train_eval_model(list_of_model, prefix_model_name, X_train, y_train, data_configuration_name, X_valid, y_valid, log_path):

    """
    This function trains and evaluates a list of machine learning models, logs their performance, 
    and returns the updated training log and list of models.

    :param list_of_model: <list> list of dictionaries containing model name, model object, and other information
    :param prefix_model_name: <str> prefix to be added to the model name
    :param X_train: <pandas DataFrame> training dataset features
    :param y_train: <pandas DataFrame> training dataset labels
    :param data_configuration_name: <str> name of the data configuration used for training
    :param X_valid: <pandas DataFrame> validation dataset features
    :param y_valid: <pandas DataFrame> validation dataset labels
    :param log_path: <str> path to the training log file
    :return training_log: <list> updated training log
    :return list_of_model: <list> list of trained models
    """
    
    list_of_model = copy.deepcopy(list_of_model)
    logger = create_log_template()

    for model in tqdm(list_of_model):    
        model_name = prefix_model_name + "-" + model["model_name"]

        start_time = time_stamp()
        model["model_object"].fit(X_train, y_train)
        finished_time = time_stamp()

        elapsed_time = finished_time - start_time
        elapsed_time = elapsed_time.total_seconds()

        y_pred = model["model_object"].predict(X_valid)
        performance = classification_report(y_valid, y_pred, output_dict = True)

        plain_id = str(start_time) + str(finished_time)
        chiper_id = hashlib.md5(plain_id.encode()).hexdigest()

        model["model_uid"] = chiper_id

        logger["model_name"].append(model_name)
        logger["model_uid"].append(chiper_id)
        logger["training_time"].append(elapsed_time)
        logger["training_date"].append(str(start_time))
        logger["performance"].append(performance)
        logger["f1_score_avg"].append(performance["macro avg"]["f1-score"])
        logger["data_configurations"].append(data_configuration_name)

    training_log = training_log_updater(logger, log_path)

    return training_log, list_of_model

# Checking performance using different model on a normal condition
training_log, list_of_model_nor = train_eval_model(list_of_model["normal"], 
                                                   "baseline_model",
                                                   X_train,
                                                   y_train,
                                                   "normal",
                                                   X_valid,
                                                   y_valid,
                                                   "app/log/training_log.json")

# Update list of model
list_of_model["normal"] = copy.deepcopy(list_of_model_nor)

# Checking performance using different model after undersampling
training_log, list_of_model_rus = train_eval_model(list_of_model["undersampling"], 
                                                   "baseline_model",
                                                   X_rus,
                                                   y_rus,
                                                   "undersampling",
                                                   X_valid,
                                                   y_valid,
                                                   "app/log/training_log.json")

# Update list of model
list_of_model["undersampling"] = copy.deepcopy(list_of_model_rus)

# Checking performance using different model after oversampling
training_log, list_of_model_ros = train_eval_model(list_of_model["oversampling"], 
                                                   "baseline_model",
                                                   X_ros,
                                                   y_ros,
                                                   "oversampling",
                                                   X_valid,
                                                   y_valid,
                                                   "app/log/training_log.json")

# Update list of model
list_of_model["oversampling"] = copy.deepcopy(list_of_model_ros)

# Checking performance using different model after SMOTE
training_log, list_of_model_sm = train_eval_model(list_of_model["smote"], 
                                                  "baseline_model",
                                                  X_sm,
                                                  y_sm,
                                                  "smote",
                                                  X_valid,
                                                  y_valid,
                                                  "app/log/training_log.json")

# Update list of model
list_of_model["smote"] = copy.deepcopy(list_of_model_sm)

def training_log_to_df(training_log):
    
    """
    This function takes in the training log and returns a pandas dataframe containing
    all training logs sorted based on f1-score and training time.

    :param training_log: <list> list of training logs
    :return training_res: <pandas dataframe> dataframe containing training logs
    """
    
    training_res = pd.DataFrame()

    for log in tqdm(training_log):
        training_res = pd.concat([training_res, pd.DataFrame(log)])
    
    training_res.sort_values(["f1_score_avg", "training_time"], ascending = [False, True], inplace = True)
    training_res.reset_index(inplace = True, drop = True)
    
    return training_res

# Store training log to dataframe
training_res = training_log_to_df(training_log)

# Show dataframe
training_res

def get_best_model(training_log_df, list_of_model):

    """
    This function takes in the training log dataframe and list of model objects and
    returns the best model object based on f1-score and training time.
    
    :param training_log_df: <pandas dataframe> dataframe containing training logs
    :param list_of_model: <dict> dictionary containing list of model objects for each data configuration
    :return model_object: <object> the best performing model object
    :raise RuntimeError: If the best model is not found in the list of model objects
    """
    
    model_object = None

    best_model_info = training_log_df.sort_values(["f1_score_avg", "training_time"], ascending = [False, True]).iloc[0]
    
    for configuration_data in list_of_model:
        for model_data in list_of_model[configuration_data]:
            if model_data["model_uid"] == best_model_info["model_uid"]:
                model_object = model_data["model_object"]
                break
    
    if model_object == None:
        raise RuntimeError("The best model not found in your list of model.")
    
    return model_object

# Get the best model object
model = get_best_model(training_res, list_of_model)

# Show best model
model

# Store hyperparameter into dictionary
dist_params_knn = {
    "algorithm" : ["ball_tree", "kd_tree", "brute"],
    "n_neighbors" : [2, 3, 4, 5, 6, 10, 15, 20, 25],
    "leaf_size" : [2, 3, 4, 5, 6, 10, 15, 20, 25],
}

# Find best hyperparameter
knn_enhance = GridSearchCV(KNeighborsClassifier(), dist_params_knn, n_jobs = -1, verbose = 420, cv = 5)

# Update list of model
list_of_model["normal"].append({"model_name": knn_enhance.__class__.__name__ + "-" + knn_enhance.estimator.__class__.__name__, "model_object": copy.deepcopy(knn_enhance), "model_uid": ""})
list_of_model["undersampling"].append({"model_name": knn_enhance.__class__.__name__ + "-" + knn_enhance.estimator.__class__.__name__, "model_object": copy.deepcopy(knn_enhance), "model_uid": ""})
list_of_model["oversampling"].append({"model_name": knn_enhance.__class__.__name__ + "-" + knn_enhance.estimator.__class__.__name__, "model_object": copy.deepcopy(knn_enhance), "model_uid": ""})
list_of_model["smote"].append({"model_name": knn_enhance.__class__.__name__ + "-" + knn_enhance.estimator.__class__.__name__, "model_object": copy.deepcopy(knn_enhance), "model_uid": ""})

# Checking performance using different model in a normal condition after hyperparameter tuning
training_log, list_of_model_nor_enhanced = train_eval_model([list_of_model["normal"][-1]],
                                                            "hyperparams",
                                                            X_train,
                                                            y_train,
                                                            "normal",
                                                            X_valid,
                                                            y_valid,
                                                            "app/log/training_log.json")
# Update list of model
list_of_model["normal"][-1] = copy.deepcopy(list_of_model_nor_enhanced[0])

# Checking performance using different model after undersampling and hyperparameter tuning
training_log, list_of_model_rus_enhanced = train_eval_model([list_of_model["undersampling"][-1]],
                                                            "hyperparams",
                                                            X_rus,
                                                            y_rus,
                                                            "undersampling",
                                                            X_valid,
                                                            y_valid,
                                                            "app/log/training_log.json")

# Update list of model
list_of_model["undersampling"][-1] = copy.deepcopy(list_of_model_rus_enhanced[0])

# Checking performance using different model after oversampling and hyperparameter tuning
training_log, list_of_model_ros_enhanced = train_eval_model([list_of_model["oversampling"][-1]],
                                                            "hyperparams",
                                                            X_ros,
                                                            y_ros,
                                                            "oversampling",
                                                            X_valid,
                                                            y_valid,
                                                            "app/log/training_log.json")

# Update list of model
list_of_model["oversampling"][-1] = copy.deepcopy(list_of_model_ros_enhanced[0])

# Checking performance using different model after smote and hyperparameter tuning
training_log, list_of_model_sm_enhanced = train_eval_model([list_of_model["smote"][-1]],
                                                           "hyperparams",
                                                           X_sm,
                                                           y_sm,
                                                           "smote",
                                                           X_valid,
                                                           y_valid,
                                                           "app/log/training_log.json")

# Update list of model
list_of_model["smote"][-1] = copy.deepcopy(list_of_model_sm_enhanced[0])

# Store training log to dataframe
training_log_to_df(training_log)

# Prediction using input
y_pred_valid = model.predict(X_valid)

# Show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_valid, y_pred_valid)

# Show classification report
print(classification_report(y_true = y_valid,
                            y_pred = y_pred_valid,
                            target_names = ["0", "1"]))

# Prediction using input
y_pred_test = model.predict(X_test)

# Show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)

# Show classification report
print(classification_report(y_true = y_test,
                            y_pred = y_pred_test,
                            target_names = ["0", "1"]))