"""Validated API request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_MUSHROOM = {
    "cap-shape": "x",
    "cap-surface": "s",
    "cap-color": "n",
    "bruises": "t",
    "odor": "p",
    "gill-attachment": "f",
    "gill-spacing": "c",
    "gill-size": "n",
    "gill-color": "k",
    "stalk-shape": "e",
    "stalk-root": "e",
    "stalk-surface-above-ring": "s",
    "stalk-surface-below-ring": "s",
    "stalk-color-above-ring": "w",
    "stalk-color-below-ring": "w",
    "veil-type": "p",
    "veil-color": "w",
    "ring-number": "o",
    "ring-type": "p",
    "spore-print-color": "k",
    "population": "s",
    "habitat": "u",
}


class MushroomFeatures(BaseModel):
    """The 22 categorical features used by the trained model."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={"examples": [EXAMPLE_MUSHROOM]},
    )

    cap_shape: Literal["b", "c", "f", "k", "s", "x"] = Field(alias="cap-shape")
    cap_surface: Literal["f", "g", "s", "y"] = Field(alias="cap-surface")
    cap_color: Literal["b", "c", "e", "g", "n", "p", "r", "u", "w", "y"] = Field(
        alias="cap-color"
    )
    bruises: Literal["f", "t"]
    odor: Literal["a", "c", "f", "l", "m", "n", "p", "s", "y"]
    gill_attachment: Literal["a", "f"] = Field(alias="gill-attachment")
    gill_spacing: Literal["c", "w"] = Field(alias="gill-spacing")
    gill_size: Literal["b", "n"] = Field(alias="gill-size")
    gill_color: Literal["b", "e", "g", "h", "k", "n", "o", "p", "r", "u", "w", "y"] = Field(
        alias="gill-color"
    )
    stalk_shape: Literal["e", "t"] = Field(alias="stalk-shape")
    stalk_root: Literal["?", "b", "c", "e", "r"] = Field(alias="stalk-root")
    stalk_surface_above_ring: Literal["f", "k", "s", "y"] = Field(
        alias="stalk-surface-above-ring"
    )
    stalk_surface_below_ring: Literal["f", "k", "s", "y"] = Field(
        alias="stalk-surface-below-ring"
    )
    stalk_color_above_ring: Literal["b", "c", "e", "g", "n", "o", "p", "w", "y"] = Field(
        alias="stalk-color-above-ring"
    )
    stalk_color_below_ring: Literal["b", "c", "e", "g", "n", "o", "p", "w", "y"] = Field(
        alias="stalk-color-below-ring"
    )
    veil_type: Literal["p"] = Field(alias="veil-type")
    veil_color: Literal["n", "o", "w", "y"] = Field(alias="veil-color")
    ring_number: Literal["n", "o", "t"] = Field(alias="ring-number")
    ring_type: Literal["e", "f", "l", "n", "p"] = Field(alias="ring-type")
    spore_print_color: Literal["b", "h", "k", "n", "o", "r", "u", "w", "y"] = Field(
        alias="spore-print-color"
    )
    population: Literal["a", "c", "n", "s", "v", "y"]
    habitat: Literal["d", "g", "l", "m", "p", "u", "w"]


class MushroomPrediction(BaseModel):
    prediction: Literal["edible", "poisonous"]
    is_poisonous: bool
    poisonous_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelInfo(BaseModel):
    name: Literal["mushroom"]
    algorithm: str
    features: list[str]
    threshold: float
    metrics: dict[str, float]
    trained_at: str
    sklearn_version: str
