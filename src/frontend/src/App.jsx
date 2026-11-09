import { useState } from "react";

const initial = {
  sepal_length: 5.1,
  sepal_width: 3.5,
  petal_length: 1.4,
  petal_width: 0.2
};

export default function App() {
  const [form, setForm] = useState(initial);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const onChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: Number(value) }));
  };

  const predict = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <main style={{ fontFamily: "Arial", maxWidth: 560, margin: "2rem auto" }}>
      <h1>Iris Classifier (scikit-learn)</h1>
      <p>Use flower measurements to predict species in real-time.</p>

      <form onSubmit={predict} style={{ display: "grid", gap: 10 }}>
        {Object.keys(form).map((k) => (
          <label key={k} style={{ display: "grid", gap: 4 }}>
            {k.replace("_", " ")}
            <input
              type="number"
              step="0.1"
              value={form[k]}
              onChange={(e) => onChange(k, e.target.value)}
              required
            />
          </label>
        ))}
        <button type="submit">Predict</button>
      </form>

      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}

      {result && (
        <section style={{ marginTop: 16 }}>
          <h3>Prediction: {result.prediction_label}</h3>
          <pre>{JSON.stringify(result.probabilities, null, 2)}</pre>
        </section>
      )}
    </main>
  );
}