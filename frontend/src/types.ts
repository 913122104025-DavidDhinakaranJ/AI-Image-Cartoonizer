export type Variant = "baseline" | "improved";

export interface StyleOption {
  id: string;
  name: string;
  preview: string;
}

export interface MetricPayload {
  edge_ssim: number;
  artifact_score: number;
}

export interface CartoonizeResponse {
  result_url: string;
  style_id: string;
  variant: Variant;
  latency_ms: number;
  metrics: MetricPayload;
}
