from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = MODEL_DIR / "iris_pipeline.joblib"
META_PATH = MODEL_DIR / "meta.joblib"


def train():
    # Realistic best practice:
    # 1) Keep preprocessing + model in one Pipeline (prevents train/serve mismatch)
    # 2) Use train/test split with stratification
    # 3) Add cross-validation for robust signal
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"CV accuracy: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\nTest classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(
        {"feature_names": feature_names, "target_names": target_names},
        META_PATH,
    )
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")


if __name__ == "__main__":
    train()