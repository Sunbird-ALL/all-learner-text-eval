import base64
import io
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from utils import denoise_with_rnnoise, get_error_arrays, get_pause_count, split_into_phonemes, processLP
from schemas import TextData, audioData, PhonemesRequest, PhonemesResponse, ErrorArraysResponse, AudioProcessingResponse
from typing import List
import jiwer
import eng_to_ipa as p

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post('/getTextMatrices', response_model=ErrorArraysResponse, summary="Compute Text Matrices", description="Computes WER, CER, insertion, deletion, substitution, confidence char list, missing char list, construct text", responses={
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {"detail": "Reference text must be provided."}
            }
        }
    },
    422: {
        "description": "Unprocessable Entity",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Unexpected error: Error processing characters: <error_message>"}
            }
        }
    }
})
async def compute_errors(data: TextData):
    try:
        # Validate input data
        if not data.reference:
            raise HTTPException(status_code=400, detail="Reference text must be provided.")

        reference = data.reference
        hypothesis = data.hypothesis if data.hypothesis is not None else ""
        language = data.language


        # Validate language
        allowed_languages = {"en", "ta", "te", "kn", "hi", "gu", "or"}
        print("language ::", language)
        if language not in allowed_languages:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}. Supported languages are: {', '.join(allowed_languages)}")

        # Process character-level differences
        try:
            charOut = jiwer.process_characters(reference, hypothesis)
        except Exception as e:
            logger.error(f"Error processing characters: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing characters: {str(e)}")

        # Compute WER
        try:
            wer = jiwer.wer(reference, hypothesis)
        except Exception as e:
            logger.error(f"Error computing WER: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error computing WER: {str(e)}")

        confidence_char_list = []
        missing_char_list = []
        construct_text = ""

        if language == "en":
            try:
                confidence_char_list, missing_char_list, construct_text = processLP(reference, hypothesis)
            except Exception as e:
                logger.error(f"Error processing LP: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing LP: {str(e)}")

        # Extract error arrays
        try:
            error_arrays = get_error_arrays(charOut.alignments, reference, hypothesis)
        except Exception as e:
            logger.error(f"Error extracting error arrays: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting error arrays: {str(e)}")

        return {
            "wer": wer,
            "cer": charOut.cer,
            "insertion": error_arrays['insertion'],
            "insertion_count": len(error_arrays['insertion']),
            "deletion": error_arrays['deletion'],
            "deletion_count": len(error_arrays['deletion']),
            "substitution": error_arrays['substitution'],
            "substitution_count": len(error_arrays['substitution']),
            "confidence_char_list": confidence_char_list,
            "missing_char_list": missing_char_list,
            "construct_text": construct_text
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@router.post("/getPhonemes", response_model=PhonemesResponse, summary="Get Phonemes", description="Converts text into phonemes.", responses={
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {"detail": "Input text cannot be empty."}
            }
        }
    },
    422: {
        "description": "Unprocessable Entity",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Unexpected error: Error getting phonemes: <error_message>"}
            }
        }
    }
})
async def get_phonemes(data: PhonemesRequest):
    try:
        if not data.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        phonemesList = split_into_phonemes(p.convert(data.text))
        return {"phonemes": phonemesList}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting phonemes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting phonemes: {str(e)}")
    
@router.post('/audio_processing', response_model=AudioProcessingResponse, summary="Process Audio", description="Processes audio by denoising and detecting pauses.", responses={
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {"detail": "Base64 string of audio must be provided."}
            }
        }
    },
    422: {
        "description": "Unprocessable Entity",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Unexpected error: <error_message>"}
            }
        }
    }
})
async def audio_processing(data: audioData):
    try:
        # Validate input data
        if not data.base64_string:
            raise HTTPException(status_code=400, detail="Base64 string of audio must be provided.")
        if not data.contentType:
            raise HTTPException(status_code=400, detail="Content type must be specified.")
        
        try:
            audio_data = data.base64_string
            audio_bytes = base64.b64decode(audio_data)
            audio_io = io.BytesIO(audio_bytes)
        except Exception as e:
            logger.error(f"Invalid base64 string: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 string: {str(e)}")

        pause_count = 0
        denoised_audio_base64 = ""

        if data.enablePauseCount:
            try:
                pause_count = get_pause_count(audio_io)
                if pause_count is None:
                    logger.error("Error during pause count detection")
                    raise HTTPException(status_code=500, detail="Error during pause count detection")
            except Exception as e:
                logger.error(f"Error during pause count detection: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error during pause count detection: {str(e)}")

        if data.enableDenoiser:
            try:
                denoised_audio_base64 = denoise_with_rnnoise(audio_data, data.contentType)
                if denoised_audio_base64 is None:
                    logger.error("Error during audio denoising")
                    raise HTTPException(status_code=500, detail="Error during audio denoising")
            except ValueError as e:
                logger.error(f"Value error in denoise_with_rnnoise: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Value error in denoise_with_rnnoise: {str(e)}")
            except RuntimeError as e:
                logger.error(f"Runtime error in denoise_with_rnnoise: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Runtime error in denoise_with_rnnoise: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in denoise_with_rnnoise: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Unexpected error in denoise_with_rnnoise: {str(e)}")

        return {
            "denoised_audio_base64": denoised_audio_base64,
            "pause_count": pause_count
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
