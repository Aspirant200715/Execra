import os
import joblib
import numpy as np

from sklearn.ensemble import IsolationForest

class TraceFeatureExtractor:

    @staticmethod
    def extract(trace_log):
        total_calls = 0
        total_lines = 0
        exception_count = 0
        max_recursion = 0
        functions = set()

        for event in trace_log:
            if event.get("event") == "call":
                total_calls += 1

            if "line" in event:
                total_lines += 1

            if event.get("exception"):
                exception_count += 1

            depth = event.get("depth", 0)
            max_recursion = max(max_recursion, depth)

            func = event.get("function")
            if func:
                functions.add(func)

        unique_functions = len(functions)

        avg_line_density = (
            total_lines / total_calls
            if total_calls > 0 else 0
        )

        return np.array([
            total_calls,
            total_lines,
            exception_count,
            max_recursion,
            unique_functions,
            avg_line_density
        ])
class TraceAnomalyDetector:

    MODEL_PATH = "models/anomaly/trace_model.pkl"

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

    def fit(self, trace_logs):
        features = np.array([
            TraceFeatureExtractor.extract(log)
            for log in trace_logs
        ])

        self.model.fit(features)

        os.makedirs(
            os.path.dirname(self.MODEL_PATH),
            exist_ok=True
        )

        joblib.dump(self.model, self.MODEL_PATH)

    def load_model(self):
        if os.path.exists(self.MODEL_PATH):
            self.model = joblib.load(self.MODEL_PATH)

    def predict(self, trace_log):
        features = TraceFeatureExtractor.extract(trace_log)

        prediction = self.model.predict([features])[0]
        score = self.model.decision_function([features])[0]

        is_anomalous = bool(prediction == -1)

        return is_anomalous, float(score)

    def analyze_trace(self, trace_log):
        try:
            self.load_model()

            is_anomalous, score = self.predict(trace_log)

            if is_anomalous:
                return {
                    "warning": (
                        "⚠️ Anomalous execution trace detected."
                    ),
                    "score": score
                }

            return {
                "warning": None,
                "score": score
            }

        except Exception as e:
            return {
                "warning": str(e),
                "score": None
            }