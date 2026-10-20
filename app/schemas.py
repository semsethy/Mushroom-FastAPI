"""Dataset category codes are the API contract; hyphenated names match the CSV."""
from typing import Literal
from pydantic import BaseModel, Field

class Mushroom(BaseModel):
    model_config = {'extra': 'forbid', 'json_schema_extra': {'examples': [{'cap-shape': 'x', 'cap-surface': 's', 'cap-color': 'n', 'bruises': 't', 'odor': 'p', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'n', 'gill-color': 'k', 'stalk-shape': 'e', 'stalk-root': 'e', 'stalk-surface-above-ring': 's', 'stalk-surface-below-ring': 's', 'stalk-color-above-ring': 'w', 'stalk-color-below-ring': 'w', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'p', 'spore-print-color': 'k', 'population': 's', 'habitat': 'u'}]}}
    cap_shape: Literal['b', 'c', 'f', 'k', 's', 'x'] = Field(alias='cap-shape')
    cap_surface: Literal['f', 'g', 's', 'y'] = Field(alias='cap-surface')
    cap_color: Literal['b', 'c', 'e', 'g', 'n', 'p', 'r', 'u', 'w', 'y'] = Field(alias='cap-color')
    bruises: Literal['f', 't'] = Field(alias='bruises')
    odor: Literal['a', 'c', 'f', 'l', 'm', 'n', 'p', 's', 'y'] = Field(alias='odor')
    gill_attachment: Literal['a', 'f'] = Field(alias='gill-attachment')
    gill_spacing: Literal['c', 'w'] = Field(alias='gill-spacing')
    gill_size: Literal['b', 'n'] = Field(alias='gill-size')
    gill_color: Literal['b', 'e', 'g', 'h', 'k', 'n', 'o', 'p', 'r', 'u', 'w', 'y'] = Field(alias='gill-color')
    stalk_shape: Literal['e', 't'] = Field(alias='stalk-shape')
    stalk_root: Literal['?', 'b', 'c', 'e', 'r'] = Field(alias='stalk-root')
    stalk_surface_above_ring: Literal['f', 'k', 's', 'y'] = Field(alias='stalk-surface-above-ring')
    stalk_surface_below_ring: Literal['f', 'k', 's', 'y'] = Field(alias='stalk-surface-below-ring')
    stalk_color_above_ring: Literal['b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'] = Field(alias='stalk-color-above-ring')
    stalk_color_below_ring: Literal['b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'] = Field(alias='stalk-color-below-ring')
    veil_type: Literal['p'] = Field(alias='veil-type')
    veil_color: Literal['n', 'o', 'w', 'y'] = Field(alias='veil-color')
    ring_number: Literal['n', 'o', 't'] = Field(alias='ring-number')
    ring_type: Literal['e', 'f', 'l', 'n', 'p'] = Field(alias='ring-type')
    spore_print_color: Literal['b', 'h', 'k', 'n', 'o', 'r', 'u', 'w', 'y'] = Field(alias='spore-print-color')
    population: Literal['a', 'c', 'n', 's', 'v', 'y'] = Field(alias='population')
    habitat: Literal['d', 'g', 'l', 'm', 'p', 'u', 'w'] = Field(alias='habitat')

class MushroomPrediction(BaseModel):
    label: Literal['edible', 'poisonous']
    poisonous_probability: float = Field(ge=0, le=1)
    will_be_poisonous: bool
    threshold: float = Field(ge=0, le=1)

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    threshold: float | None = None
    features: list[str] = Field(default_factory=list)
    sklearn_version: str | None = None
