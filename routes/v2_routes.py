from fastapi import APIRouter
from routes.common_routes import compute_errors, get_phonemes, audio_processing, upload_audio

v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

# Reuse the shared API
v2_router.add_api_route('/getTextMatrices', compute_errors, methods=["POST"])

v2_router.add_api_route('/getPhonemes', get_phonemes, methods=["POST"])

v2_router.add_api_route('/audio_processing', audio_processing, methods=["POST"])

v2_router.add_api_route('/uploadAudio', upload_audio, methods=["POST"])