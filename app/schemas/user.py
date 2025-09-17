from pydantic import BaseModel, Field, constr
from datetime import datetime

Username = constr(strip_whitespace=True, min_length=1, max_length=20)

class UserCreate(BaseModel):
    username: Username
    total_score: int = Field(ge=0)

class UserOut(BaseModel):
    user_id: int
    username: str
    total_score: int
    created_at: datetime

    class Config:
        from_attributes = True
