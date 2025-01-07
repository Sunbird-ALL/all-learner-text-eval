from pydantic import BaseModel,Field
from typing import List, Optional, Dict 

class TextData(BaseModel):
    reference: str = Field(..., example="frog jumps", description="The reference text to compare against.")
    hypothesis: Optional[str] = Field(None, example="dog jumps", description="The hypothesis text to be compared.")
    language: str = Field(..., example="en", description="The language of the text.")

class audioData(BaseModel):
    base64_string: str = Field(..., example="UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAABCxAgAEABAAZGF0YUAA", description="Base64 encoded audio string.")
    enablePauseCount: bool = Field(..., example=True, description="Flag to enable pause count detection.")
    enableDenoiser: bool = Field(..., example=True, description="Flag to enable audio denoising.")
    contentType: str = Field(..., example="Word", description="The type of content in the audio.")
    language: str = Field(..., example="en", description="language of the audio.")

class PhonemesRequest(BaseModel):
    text: str = Field(..., example="dog jumps", description="The text to convert into phonemes.")

class PhonemesResponse(BaseModel):
    phonemes: List[str] = Field(..., example=["d", "ɔ", "g", "ʤ", "ə", "m", "p", "s"], description="List of phonemes extracted from the text.")

class ErrorArraysResponse(BaseModel):
    wer: float = Field(..., example=0.5, description="Word Error Rate.")
    cer: float = Field(..., example=0.2, description="Character Error Rate.")
    insertion: List[str] = Field(..., example=[], description="List of insertions.")
    insertion_count: int = Field(..., example=0, description="Count of insertions.")
    deletion: List[str] = Field(..., example=["r"], description="List of deletions.")
    deletion_count: int = Field(..., example=1, description="Count of deletions.")
    substitution: List[Dict[str, str]] = Field(..., example=[{"removed": "d", "replaced": "f"}], description="List of substitutions.")
    substitution_count: int = Field(..., example=1, description="Count of substitutions.")
    pause_count: Optional[int] = Field(None, example=None, description="Count of pauses detected.")
    confidence_char_list: Optional[List[str]] = Field(None, example=["p", "ʤ", "s", "ə", "m"], description="List of characters with confidence levels.")
    missing_char_list: Optional[List[str]] = Field(None, example=["f", "g", "r", "ɑ"], description="List of missing characters.")
    construct_text: Optional[str] = Field(None, example="jumps", description="Constructed text based on the hypothesis.")

class AudioProcessingResponse(BaseModel):
    concatenated_audio_base64: str = Field(..., example="UkiGRV////wqgwbwrbw////AAAA", description="Base64 encoded denoised audio.")
    denoised_audio_base64: str = Field(..., example="UkiGRV////wqgwbwrbw////AAAA", description="Base64 encoded denoised audio.")
    pause_count: Optional[int] = Field(..., example=2, description="Count of pauses detected.")