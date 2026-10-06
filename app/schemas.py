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
    token_count: int = 0
    segments: list[TranscriptSegment] = Field(
        default_factory=list
    )

class Evidence(BaseModel):
    text: str
    start: float
    end: Optional[float] = None

class MetricResult(BaseModel):
    precision: float
    recall: float
    f1: float


class EvaluationReport(BaseModel):
    action_items: MetricResult
    decisions: MetricResult
    evidence_validity: float

class GroundTruthItem(BaseModel):
    text: str

class JudgeResult(BaseModel):
    match: bool
    score: float
    reason: str

class GroundTruthMeeting(BaseModel):
    action_items: list[GroundTruthItem] = Field(
        default_factory=list
    )
    decisions: list[GroundTruthItem] = Field(
        default_factory=list
    )

class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    evidence: Optional[Evidence] = None

class Decision(BaseModel):
    description: str
    evidence: Optional[Evidence] = None

class ChunkAnalysis(BaseModel):
    key_points: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


class MeetingMinutes(BaseModel):
    title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)