import io
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from pydub import AudioSegment
from pydub.silence import detect_silence
import jiwer

app = Flask(__name__)

def get_error_arrays(alignments, reference, hypothesis, base64string):
    insertion = []
    deletion = []
    substitution = []

    for chunk in alignments[0]:
        if chunk.type == 'insert':
            insertion.extend(
                list(range(chunk.hyp_start_idx, chunk.hyp_end_idx)))
        elif chunk.type == 'delete':
            deletion.extend(
                list(range(chunk.ref_start_idx, chunk.ref_end_idx)))
        elif chunk.type == 'substitute':
            refslice = slice(chunk.ref_start_idx, chunk.ref_end_idx)
            hyposlice = slice(chunk.hyp_start_idx, chunk.hyp_end_idx)

            substitution.append({
                "removed": hypothesis[hyposlice],
                "replaced": reference[refslice]
            })

    insertion_chars = [hypothesis[i] for i in insertion]
    deletion_chars = [reference[i] for i in deletion]

    # For count the pauses in audio files
    audio_data = base64.b64decode(base64string)

    # Use pydub to load the audio from the BytesIO object
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

    # Check if the audio is completely silent or empty
    silence_ranges = detect_silence(
        audio_segment, min_silence_len=100, silence_thresh=-40)

    if len(silence_ranges) == 1 and silence_ranges[0] == [0, len(audio_segment)]:
        pause_count = 0
    else:
        # Count pause occurrences
        pause_count = len(silence_ranges)
    
    return {
        'insertion': insertion_chars,
        'deletion': deletion_chars,
        'substitution': substitution,
        'pause_count': pause_count
    }


@app.route('/getTextMatrices', methods=['POST'])
def compute_errors():
    data = request.get_json()
    reference = data.get('reference')
    hypothesis = data.get('hypothesis')
    base64_string = data.get('base64_string')

    charOut = jiwer.process_characters(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis)

    # Extract error arrays
    error_arrays = get_error_arrays(
        charOut.alignments, reference, hypothesis, base64_string)

    return jsonify({
        "wer": wer,
        "cer": charOut.cer,
        "insertion": error_arrays['insertion'],
        "insertion_count": len(error_arrays['insertion']),
        "deletion": error_arrays['deletion'],
        "deletion_count": len(error_arrays['deletion']),
        "substitution": error_arrays['substitution'],
        "substitution_count": len(error_arrays['substitution']),
        "pause_count": error_arrays['pause_count']
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
