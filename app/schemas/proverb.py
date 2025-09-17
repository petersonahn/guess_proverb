from pydantic import BaseModel

class ProverbOut(BaseModel):
    id: int
    question: str
    hint: str | None = None

    class Config:
        from_attributes = True
