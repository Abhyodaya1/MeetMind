from pydantic import BaseModel, Field
from typing import Optional

class SpeakerTurn(BaseModel):
    speaker: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: Optional[float] = None
    speaker: Optional[str] = None

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


class IndexedChunk(BaseModel):
    meeting_id: str
    chunk_index: int
    text: str
    start: float
    end: Optional[float] = None
    speakers: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk: IndexedChunk
    score: float


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[RetrievedChunk] = Field(default_factory=list)


class VisualMaterial(BaseModel):
    source_path: str
    material_type: str = "slide"
    timestamp: Optional[float] = None
    extracted_text: str = ""
    summary: str = ""


class EvidenceValidationSummary(BaseModel):
    action_items_valid: int
    action_items_total: int
    decisions_valid: int
    decisions_total: int
    validity_ratio: float


class MeetingProcessingResult(BaseModel):
    minutes: MeetingMinutes
    segments: list[TranscriptSegment]
    chunks: list[TranscriptChunk]
    visual_materials: list[VisualMaterial] = Field(default_factory=list)
    validation_summary: EvidenceValidationSummary