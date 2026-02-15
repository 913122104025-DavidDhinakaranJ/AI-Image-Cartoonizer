import type { CartoonizeResponse } from "../types";

interface ResultPanelProps {
  baseline: CartoonizeResponse;
  improved: CartoonizeResponse;
  improvedUrl: string;
}

export function ResultPanel({ baseline, improved, improvedUrl }: ResultPanelProps) {
  return (
    <section className="panel metrics">
      <h2>Comparison Metrics</h2>
      <div className="metric-grid">
        <div>
          <p className="metric-name">Edge SSIM</p>
          <p>{baseline.metrics.edge_ssim} (baseline)</p>
          <p>{improved.metrics.edge_ssim} (improved)</p>
        </div>
        <div>
          <p className="metric-name">Artifact Score</p>
          <p>{baseline.metrics.artifact_score} (baseline)</p>
          <p>{improved.metrics.artifact_score} (improved)</p>
        </div>
        <div>
          <p className="metric-name">Latency</p>
          <p>{baseline.latency_ms} ms (baseline)</p>
          <p>{improved.latency_ms} ms (improved)</p>
        </div>
      </div>
      <a className="download-link" href={improvedUrl} download>
        Download Improved Result
      </a>
    </section>
  );
}
