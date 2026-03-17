import { useEffect, useState } from "react";

import { cartoonize, fetchStyles, resolveResultUrl } from "./api/client";
import { ComparisonView } from "./components/ComparisonView";
import { ImageUploader } from "./components/ImageUploader";
import { ResultPanel } from "./components/ResultPanel";
import { StyleSelector } from "./components/StyleSelector";
import type { CartoonizeResponse, StyleOption } from "./types";

function App() {
  const [styles, setStyles] = useState<StyleOption[]>([]);
  const [selectedStyleId, setSelectedStyleId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [baseline, setBaseline] = useState<CartoonizeResponse | null>(null);
  const [improved, setImproved] = useState<CartoonizeResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStyles()
      .then((items) => {
        setStyles(items);
        if (items.length > 0) {
          setSelectedStyleId(items[0].id);
        }
      })
      .catch((err) => setError((err as Error).message));
  }, []);

  const [inputPreviewUrl, setInputPreviewUrl] = useState("");

  useEffect(() => {
    if (!selectedFile) {
      setInputPreviewUrl("");
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setInputPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const baselineUrl = baseline ? resolveResultUrl(baseline.result_url) : "";
  const improvedUrl = improved ? resolveResultUrl(improved.result_url) : "";

  const onRunComparison = async () => {
    if (!selectedFile || !selectedStyleId) {
      setError("Please upload an image and select a style.");
      return;
    }
    setError(null);
    setIsRunning(true);
    setBaseline(null);
    setImproved(null);

    try {
      const [baselineResult, improvedResult] = await Promise.all([
        cartoonize(selectedFile, selectedStyleId, "baseline"),
        cartoonize(selectedFile, selectedStyleId, "improved_lite")
      ]);
      setBaseline(baselineResult);
      setImproved(improvedResult);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="hero">
        <p className="kicker">Model Improvement Demo</p>
        <h1>AI Image Cartoonizer</h1>
        <p>
          Baseline AnimeGAN-style inference vs AQE-enhanced output with measurable
          metrics.
        </p>
      </header>

      <section className="controls">
        <ImageUploader onSelectFile={setSelectedFile} />
        <StyleSelector
          styles={styles}
          selectedStyleId={selectedStyleId}
          onSelect={setSelectedStyleId}
        />
      </section>

      <section className="panel action-panel">
        <button
          type="button"
          className="run-btn"
          onClick={onRunComparison}
          disabled={isRunning || !selectedFile || !selectedStyleId}
        >
          {isRunning ? "Running..." : "Generate Baseline + Improved Lite"}
        </button>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {baseline && improved && inputPreviewUrl ? (
        <>
          <ComparisonView
            inputPreviewUrl={inputPreviewUrl}
            baselineUrl={baselineUrl}
            improvedUrl={improvedUrl}
          />
          <ResultPanel baseline={baseline} improved={improved} improvedUrl={improvedUrl} />
        </>
      ) : null}
    </main>
  );
}

export default App;
