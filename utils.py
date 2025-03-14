import base64
import io
import ffmpeg
from functools import lru_cache
import eng_to_ipa as p
from fuzzywuzzy import fuzz
import soundfile as sf

english_phoneme = ["b","d","f","g","h","ʤ","k","l","m","n","p","r","s","t","v","w","z","ʒ","tʃ","ʃ","θ","ð","ŋ","j","æ","eɪ","ɛ","i:","ɪ","aɪ","ɒ","oʊ","ʊ","ʌ","u:","ɔɪ","aʊ","ə","eəʳ","ɑ:","ɜ:ʳ","ɔ:","ɪəʳ","ʊəʳ","i","u","ɔ","ɑ","ɜ","e","ʧ","o","y","a", "x", "c"]
anamoly_list = {}

def denoise_with_rnnoise(audio_base64, content_type, padding_duration=0.1, time_stretch_factor=0.75):
    try:
        # Decode base64 to get the audio data
        try:
            audio_data = base64.b64decode(audio_base64)
        except base64.binascii.Error as e:
            raise ValueError(f"Invalid base64 string: {str(e)}")

        audio_io = io.BytesIO(audio_data)
        input_audio = audio_io.read()

        # Path to the RNNoise model
        model_path = "./audio_model/cb.rnnn"

        # Create the ffmpeg filter chain
        filter_chain = []
        if content_type.lower() == 'word':
            filter_chain.append(f'apad=pad_dur={padding_duration}')
            filter_chain.append(f'apad=pad_dur={padding_duration}')
        filter_chain.append(f'atempo={time_stretch_factor}')
        filter_chain_str = ','.join(filter_chain)

        # Apply the filters and denoise
        try:
            output, _ = (
                ffmpeg
                .input('pipe:', format='wav')
                .output('pipe:', format='wav', af=f'{filter_chain_str},arnndn=m={model_path}')
                .run(input=input_audio, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error during noise reduction with FFmpeg: {e.stderr.decode()}")

        # Convert the processed output back to base64
        try:
            denoised_audio_base64 = base64.b64encode(output).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Error encoding output to base64: {str(e)}")
        
        # Clear cache to free memory
        del audio_data
        del audio_io

        return denoised_audio_base64

    except ValueError as e:
        print(f"Value error in denoise_with_rnnoise: {str(e)}")
        raise
    except RuntimeError as e:
        print(f"Runtime error in denoise_with_rnnoise: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error in denoise_with_rnnoise: {str(e)}")
        raise

def convert_to_base64(audio_data, sample_rate):
    try:
        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_data, sample_rate, format='wav')
        except Exception as e:
            raise RuntimeError(f"Error writing audio data to buffer: {str(e)}")

        buffer.seek(0)
        try:
            base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Error encoding buffer to base64: {str(e)}")

        return base64_audio
    except Exception as e:
        print(f"Error in convert_to_base64: {str(e)}")
        return {"error": str(e)}
       
def get_error_arrays(alignments, reference, hypothesis):
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

    return {
        'insertion': insertion_chars,
        'deletion': deletion_chars,
        'substitution': substitution, 
    }

def get_pause_count(audio_io):
        # Run the FFmpeg command with the input from the byte stream
        process = (
            ffmpeg
            .input('pipe:0')
            .filter('silencedetect', noise='-40dB', duration=0.5)
            .output('pipe:1', format='null')
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        # Write the audio data to the stdin of the FFmpeg process
        stdout, stderr = process.communicate(input=audio_io.read())
        # Parse the stderr output to count the silences
        silence_lines = stderr.decode().split('\n')
        silence_start_count = sum(1 for line in silence_lines if "silence_start" in line)
        return silence_start_count

def find_closest_match(target_word, input_string):
    # Tokenize the input string into words
    words = input_string.lower().split()
    targ = target_word.lower()
    # Initialize variables to keep track of the best match
    best_match = None
    best_score = 0
    
    # Iterate through the words in the input string
    for word in words:
        similarity_score = fuzz.ratio(targ, word)
        
        # Update the best match if a higher score is found
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = word
    
    return best_match, best_score

@lru_cache(maxsize=None)
def split_into_phonemes(token):
    # Phoneme mapping for combined phonemes
    combined_phonemes = {
        "dʒ": "ʤ",
        "tʃ": "ʧ",
        "ɪəʳ": "ɪəʳ",
        "ʊəʳ": "ʊəʳ",
        "eɪʳ": "eɪ",
        "aɪ": "aɪ",
        "oʊ": "o",
        "ɔɪ": "ɔɪ",
        "aʊ": "aʊ",
        "eəʳ": "eəʳ",
        "ɑ:": "ɑ",
        "ɜ:ʳ": "ɜ:ʳ",
        "ɔ:": "ɔ:",
        "i:": "i",
    }
    
    # Set of characters to skip (stress marks, etc.)
    skip_chars = {"'", " ", "ˈ", "ˌ"}

    # Convert the english_phoneme list into a set for O(1) average-time complexity checks
    english_phoneme_set = set(english_phoneme)
    
    ph_list = []
    word_list = token.split()  # split by whitespace (space, tab, newline, etc.)

    for p in word_list:
        size = len(p)
        i = 0
        while i < size:
            if p[i] in skip_chars:
                i += 1
                continue

            # Check for combined phonemes first (3 then 2 characters long)
            if i + 3 <= size and p[i:i+3] in combined_phonemes:
                ph_list.append(combined_phonemes[p[i:i+3]])
                i += 3
            elif i + 2 <= size and p[i:i+2] in combined_phonemes:
                ph_list.append(combined_phonemes[p[i:i+2]])
                i += 2
            elif i + 1 <= size and p[i:i+1] in english_phoneme_set:
                ph_list.append(p[i:i+1])
                i += 1
            else:
                # Log an anomaly if the character isn't recognized
                ph_list.append(p[i])
                if p[i] not in anamoly_list:
                    anamoly_list[p[i]] = 1
                else:
                    anamoly_list[p[i]] += 1
                i += 1

    return ph_list

def identify_missing_tokens(orig_text, resp_text):
    # Splitting text into words
    orig_word_list = orig_text.lower().split()
    resp_word_list = resp_text.lower().split()
    
    # Initialize lists and dictionaries
    construct_word_list = []
    missing_word_list = []
    orig_phoneme_list = []
    construct_phoneme_list = []
    missing_phoneme_list = []
    construct_text = []
    
    # Precompute phonemes for response words for quick lookup
    resp_phonemes = {word: p.convert(word).replace('*', '') for word in resp_word_list}
    for word in orig_word_list:
        # Precompute original word phonemes
        p_word = p.convert(word)
        
        # Find closest match based on precomputed phonemes to avoid redundant calculations
        closest_match, similarity_score = find_closest_match(word, resp_text)
        
        # Check similarity and categorize word
        if similarity_score > 80:
            construct_word_list.append(closest_match)
            p_closest_match = resp_phonemes[closest_match]
            construct_phoneme_list.append(split_into_phonemes(p_closest_match))
            construct_text.append(closest_match)
        else:
            missing_word_list.append(word)
            p_word_phonemes = split_into_phonemes(p_word)
            missing_phoneme_list.append(p_word_phonemes)
        
        # Store original phonemes for each word
        orig_phoneme_list.append(split_into_phonemes(p_word))

    # Convert list of words to a single string
    construct_text = ' '.join(construct_text)

    # Efficiently deduplicate and flatten phoneme lists
    #orig_flatList = set(phoneme for sublist in orig_phoneme_list for phoneme in sublist)
    missing_flatList = set(phoneme for sublist in missing_phoneme_list for phoneme in sublist)
    construct_flatList = set(phoneme for sublist in construct_phoneme_list for phoneme in sublist)

    return list(construct_flatList), list(missing_flatList) ,construct_text

def processLP(orig_text, resp_text):
    cons_list, miss_list, construct_text = identify_missing_tokens(orig_text, resp_text)

    #remove phonemes from miss_list which are in cons_list, ?but add those phonemes a count of could be issu
    # phonemes in constructed list are familiar ones
    # phonemes that are in miss_list and not in cons_list are the unfamiliar ones
    unfamiliar_list = []
    for c in miss_list:
        if c not in cons_list:
            unfamiliar_list.append(c)
    #function to calculate wer cer, substitutions, deletions and insertions, silence, repetitions
    #insert into DB the LearnerProfile vector
    return cons_list, miss_list,construct_text