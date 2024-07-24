from numpy import array
from pandas import read_csv as pandas_read_csv
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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


# Normalize the predictors
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameters for the XGBoost model
# param = {
#    "max_depth": 30,  # the maximum depth of each tree
#    "eta": 0.3,  # the training step for each iteration
#    "objective": "reg:squarederror",  # error evaluation for multiclass training
# }  # the number of classes that exist in this dataset

# Define the base models
level0 = list()
level0.append(("lr", LinearRegression()))
# level0.append(
#    ("xgb", XGBRegressor(objective="reg:squarederror", max_depth=30, eta=0.3))
# )

# Define meta learner model
# model = LinearRegression()
model = XGBRegressor(objective="reg:squarederror", max_depth=30, eta=0.3)

# Define the stacking ensemble
# model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)

# Fit the model on all available data
model.fit(X_train, y_train)

# Make a prediction
preds = model.predict(X_test)

print(preds)
print(y_test)

X_test2 = array([[0.75, 0.00575, 0.001650]])
X_test22 = scaler.transform(X_test2)
print(model.predict(X_test22))
