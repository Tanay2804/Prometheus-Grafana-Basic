from pydantic import BaseModel, Field, field_validator


class HeartRequest(BaseModel):
    features: list[float] = Field(..., min_length=13, max_length=13)

    @field_validator("features")
    @classmethod
    def ensure_numeric_values(cls, values: list[float]) -> list[float]:
        return [float(value) for value in values]
