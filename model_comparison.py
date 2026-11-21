from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_DIR / "Data" / "Raw" /
    "HousePricePredictionIndore.csv"
)

MODEL_DIR = PROJECT_DIR / "Models"
MODEL_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH)

X = data.drop("Price", axis=1)
y = data["Price"]

num_cols = [
    "Area",
    "BHK",
    "Floor",
    "Balconies",
    "Property Age"
]

cat_cols = [
    "Location",
    "Parking",
    "Near Market",
    "Main Road",
    "Furnishing"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        )
    ]
)

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print("\nMODEL COMPARISON")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    PROJECT_DIR / "model_comparison_results.csv",
    index=False
)

best_model_name = comparison_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)

print("\nBest Model:", best_model_name)
print("Best model saved successfully!")

graph_df = comparison_df.set_index("Model")

for metric in ["MAE", "RMSE"]:
    minimum = graph_df[metric].min()
    maximum = graph_df[metric].max()

    if maximum == minimum:
        graph_df[metric + " Score"] = 1.0
    else:
        graph_df[metric + " Score"] = (
            maximum - graph_df[metric]
        ) / (maximum - minimum)

minimum = graph_df["R2 Score"].min()
maximum = graph_df["R2 Score"].max()

if maximum == minimum:
    graph_df["R2 Performance"] = 1.0
else:
    graph_df["R2 Performance"] = (
        graph_df["R2 Score"] - minimum
    ) / (maximum - minimum)

plt.figure(figsize=(10, 6))

x = np.arange(len(graph_df))

plt.plot(
    x,
    graph_df["MAE Score"],
    marker="o",
    linewidth=2,
    label="MAE Performance"
)

plt.plot(
    x,
    graph_df["RMSE Score"],
    marker="s",
    linewidth=2,
    label="RMSE Performance"
)

plt.plot(
    x,
    graph_df["R2 Performance"],
    marker="^",
    linewidth=2,
    label="R2 Performance"
)

plt.xticks(
    x,
    graph_df.index,
    rotation=10
)

plt.ylim(-0.05, 1.05)

plt.xlabel("Machine Learning Models")
plt.ylabel("Normalized Performance Score")

plt.title("Machine Learning Model Comparison")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nComparison graph saved successfully!")
print("All tasks completed!")