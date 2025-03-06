from fastapi import APIRouter
from routes.common_routes import compute_errors, audio_processing, get_phonemes, upload_audio

v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

# Reuse the shared API
v1_router.add_api_route('/getTextMatrices', compute_errors, methods=["POST"])

v1_router.add_api_route('/getPhonemes', get_phonemes, methods=["POST"])

v1_router.add_api_route('/audio_processing', audio_processing, methods=["POST"])

v1_router.add_api_route('/uploadAudio', upload_audio, methods=["POST"])