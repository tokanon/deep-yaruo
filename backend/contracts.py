from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class TextTransportRequest(BaseModel):
    text: str = Field(max_length=2_000_000)


@dataclass(frozen=True)
class ConversionOptions:
    columns: int = 72
    font_size: int = 16
    detail: int = 55
    threshold_low: int = 45
    threshold_high: int = 135
    min_component: int = 8
    abstraction: int = 0
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_width: float = 1.0
    crop_height: float = 1.0
    max_rows: int = 64
    profile: str = "auto"

    def normalized(self) -> "ConversionOptions":
        if self.profile not in {
            "auto",
            "person",
            "background",
            "lineart",
            "background_lineart",
        }:
            raise ValueError(
                "profile must be 'auto', 'person', 'background', 'lineart', "
                "or 'background_lineart'."
            )
        x = min(max(self.crop_x, 0.0), 0.95)
        y = min(max(self.crop_y, 0.0), 0.95)
        width = min(max(self.crop_width, 0.05), 1.0 - x)
        height = min(max(self.crop_height, 0.05), 1.0 - y)
        low = min(max(self.threshold_low, 0), 254)
        high = min(max(self.threshold_high, low + 1), 255)
        return ConversionOptions(
            columns=min(max(self.columns, 24), 140),
            font_size=min(max(self.font_size, 12), 24),
            detail=min(max(self.detail, 0), 100),
            threshold_low=low,
            threshold_high=high,
            min_component=min(max(self.min_component, 0), 200),
            abstraction=min(max(self.abstraction, 0), 100),
            crop_x=x,
            crop_y=y,
            crop_width=width,
            crop_height=height,
            max_rows=min(max(self.max_rows, 12), 100),
            profile=self.profile,
        )


@dataclass
class ConversionResult:
    ascii_text: str
    rows: int
    columns: int
    processed_png: str
    rendered_png: str
    crop: tuple[int, int, int, int]
