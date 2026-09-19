import json

from app import generate_house_data, train_price_model


df = generate_house_data()
result = train_price_model(df)

summary = {
    "rows": int(len(df)),
    "columns": list(df.columns),
    "price_mean": round(float(df["price"].mean()), 2),
    "price_median": round(float(df["price"].median()), 2),
    "price_min": int(df["price"].min()),
    "price_max": int(df["price"].max()),
    "districts": sorted(df["district"].unique().tolist()),
    "mae": round(float(result["mae"]), 4),
    "r2": round(float(result["r2"]), 4),
    "coefficients": {
        name: round(float(value), 4)
        for name, value in zip(result["feature_cols"], result["model"].coef_)
    },
    "intercept": round(float(result["model"].intercept_), 4),
}

print(json.dumps(summary, ensure_ascii=False, indent=2))
