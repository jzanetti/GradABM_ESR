import xgboost as xgb
from pandas import read_csv as pandas_read_csv
from sklearn.model_selection import train_test_split

prerun_data_path = (
    "/tmp/gradabm_esr/Auckland_2019_measles3/train/prerun/prerun_stats.csv"
)

prerun_data = pandas_read_csv(prerun_data_path)

# Assuming df is your DataFrame and it has columns 'x', 'y', and 'z'
X = prerun_data[
    [
        "infection_gamma_scaling_factor",
        "vaccine_efficiency_symptom",
        "initial_infected_percentage",
    ]
]  # predictors
y = prerun_data["lost"]  # target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Convert the data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Define the parameters for the XGBoost model
param = {
    "max_depth": 30,  # the maximum depth of each tree
    "eta": 0.3,  # the training step for each iteration
    "objective": "reg:squarederror",  # error evaluation for multiclass training
}  # the number of classes that exist in this dataset

# Train the model
num_round = 100  # the number of training iterations
bst = xgb.train(param, dtrain, num_round)

# Make predictions
preds = bst.predict(dtest)

print(preds)
print(y_test)
