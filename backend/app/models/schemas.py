from typing import Literal

from pydantic import BaseModel, Field


Variant = Literal["baseline", "improved"]


class StyleInfo(BaseModel):
    id: str = Field(..., examples=["hayao"])
    name: str = Field(..., examples=["Hayao"])
    preview: str = Field(..., examples=["/static/previews/hayao.svg"])


class StylesResponse(BaseModel):
    styles: list[StyleInfo]


class MetricPayload(BaseModel):
    edge_ssim: float
    artifact_score: float


class CartoonizeResponse(BaseModel):
    result_url: str
    style_id: str
    variant: Variant
    latency_ms: int
    metrics: MetricPayload


class HealthResponse(BaseModel):
    status: str
