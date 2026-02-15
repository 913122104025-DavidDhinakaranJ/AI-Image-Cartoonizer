interface ComparisonViewProps {
  inputPreviewUrl: string;
  baselineUrl: string;
  improvedUrl: string;
}

export function ComparisonView({
  inputPreviewUrl,
  baselineUrl,
  improvedUrl
}: ComparisonViewProps) {
  return (
    <div className="compare-grid">
      <figure className="panel image-panel">
        <figcaption>Input</figcaption>
        <img src={inputPreviewUrl} alt="Input preview" />
      </figure>
      <figure className="panel image-panel">
        <figcaption>Baseline</figcaption>
        <img src={baselineUrl} alt="Baseline cartoonized result" />
      </figure>
      <figure className="panel image-panel">
        <figcaption>Improved AQE</figcaption>
        <img src={improvedUrl} alt="Improved cartoonized result" />
      </figure>
    </div>
  );
}
