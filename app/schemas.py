from pydantic import BaseModel, Field
from typing import Optional

class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: Optional[float] = None

class TranscriptChunk(BaseModel):
    text: str
    start: float
    end: Optional[float] = None


class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    evidence: Optional[str] = None

class ChunkAnalysis(BaseModel):
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


class MeetingMinutes(BaseModel):
    title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)