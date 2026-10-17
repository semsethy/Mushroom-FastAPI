"""What the API accepts and returns. FastAPI validates requests against these and documents
them at /docs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EXAMPLE_CUSTOMER = {
    "account_weeks": 98,
    "contract_renewal": 0,
    "data_plan": 0,
    "data_usage": 0.5,
    "cust_serv_calls": 5,
    "day_mins": 160.0,
    "day_calls": 102,
    "monthly_charge": 49.8,
    "overage_fee": 10.5,
    "roam_mins": 11.0,
}

EXAMPLE_FEEDBACK = "Calls drop every evening and support never calls back."


class Customer(BaseModel):
    """One customer's account and usage: the 10 features the churn model was trained on."""

    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [EXAMPLE_CUSTOMER]}}

    account_weeks: int = Field(ge=0, le=5200, description="Weeks the account has been active")
    contract_renewal: int = Field(ge=0, le=1, description="1 if they recently renewed")
    data_plan: int = Field(ge=0, le=1, description="1 if they have a data plan")
    data_usage: float = Field(ge=0, le=1000, description="Gigabytes of data per month")
    cust_serv_calls: int = Field(ge=0, le=100, description="Calls to customer service")
    day_mins: float = Field(ge=0, le=50_000, description="Daytime minutes per month")
    day_calls: int = Field(ge=0, le=10_000, description="Daytime calls per month")
    monthly_charge: float = Field(ge=0, le=10_000, description="Average monthly bill")
    overage_fee: float = Field(ge=0, le=10_000, description="Largest overage fee in 12 months")
    roam_mins: float = Field(ge=0, le=50_000, description="Roaming minutes per month")


class ChurnPrediction(BaseModel):
    churn_probability: float = Field(description="Probability that the customer leaves (0-1)")
    will_churn: bool = Field(description="churn_probability >= threshold")
    risk: Literal["low", "medium", "high"]
    threshold: float = Field(description="The decision threshold chosen during training")


class Feedback(BaseModel):
    """A piece of customer feedback: a review, a support message, a survey comment."""

    model_config = {"json_schema_extra": {"examples": [{"text": EXAMPLE_FEEDBACK}]}}

    text: str = Field(min_length=1, max_length=5000)


class SentimentPrediction(BaseModel):
    label: Literal["negative", "neutral", "positive"]
    confidence: float = Field(description="Probability of the predicted label")
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
    metrics: dict[str, float] = {}


from app.mushroom import EXAMPLE_MUSHROOM


class Mushroom(BaseModel):
    """All 22 Kaggle category codes; hyphenated keys match the CSV headers."""

    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [EXAMPLE_MUSHROOM]}}

    cap_shape: Literal['b', 'c', 'x', 'f', 'k', 's'] = Field(alias='cap-shape')
    cap_surface: Literal['f', 'g', 'y', 's'] = Field(alias='cap-surface')
    cap_color: Literal['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'] = Field(alias='cap-color')
    bruises: Literal['t', 'f'] = Field(alias='bruises')
    odor: Literal['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'] = Field(alias='odor')
    gill_attachment: Literal['a', 'd', 'f', 'n'] = Field(alias='gill-attachment')
    gill_spacing: Literal['c', 'w', 'd'] = Field(alias='gill-spacing')
    gill_size: Literal['b', 'n'] = Field(alias='gill-size')
    gill_color: Literal['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'] = Field(alias='gill-color')
    stalk_shape: Literal['e', 't'] = Field(alias='stalk-shape')
    stalk_root: Literal['b', 'c', 'u', 'e', 'z', 'r', '?'] = Field(alias='stalk-root')
    stalk_surface_above_ring: Literal['f', 'y', 'k', 's'] = Field(alias='stalk-surface-above-ring')
    stalk_surface_below_ring: Literal['f', 'y', 'k', 's'] = Field(alias='stalk-surface-below-ring')
    stalk_color_above_ring: Literal['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'] = Field(alias='stalk-color-above-ring')
    stalk_color_below_ring: Literal['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'] = Field(alias='stalk-color-below-ring')
    veil_type: Literal['p', 'u'] = Field(alias='veil-type')
    veil_color: Literal['n', 'o', 'w', 'y'] = Field(alias='veil-color')
    ring_number: Literal['n', 'o', 't'] = Field(alias='ring-number')
    ring_type: Literal['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'] = Field(alias='ring-type')
    spore_print_color: Literal['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'] = Field(alias='spore-print-color')
    population: Literal['a', 'c', 'n', 's', 'v', 'y'] = Field(alias='population')
    habitat: Literal['g', 'l', 'm', 'p', 'u', 'w', 'd'] = Field(alias='habitat')


class MushroomPrediction(BaseModel):
    label: Literal["edible", "poisonous"]
    poisonous_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
