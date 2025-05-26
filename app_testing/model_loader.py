# model_loader.py
import torch
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

# --- ASR Model Loading ---
def load_asr_model(config):
    print(f"Loading ASR model: {config.WHISPER_MODEL_SIZE} on {config.WHISPER_DEVICE} with {config.WHISPER_COMPUTE_TYPE}")
    try:
        model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            # download_root="./models_cache/faster-whisper" # Optional: specify local cache
        )
        print("ASR model loaded successfully.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load ASR model: {e}")
        raise

# --- MT Model Loading ---
def load_mt_models(config):
    print("Loading MT models...")
    mt_tokenizers = {}
    mt_models = {}
    if not hasattr(config, 'MARIANMT_MODEL_MAP') or not config.MARIANMT_MODEL_MAP:
        print("Warning: MARIANMT_MODEL_MAP is not defined or empty in config. No MT models will be loaded.")
        return mt_tokenizers, mt_models

    for key, model_name in config.MARIANMT_MODEL_MAP.items():
        try:
            print(f"Loading MT model: {model_name} for language pair: {key}")
            # tokenizer_path = f"./models_cache/marianmt/{model_name.replace('/', '_')}" # Optional local cache
            # model_path = f"./models_cache/marianmt/{model_name.replace('/', '_')}"     # Optional local cache
            tokenizer = MarianTokenizer.from_pretrained(model_name) #, cache_dir=tokenizer_path)
            model = MarianMTModel.from_pretrained(model_name).to(config.MARIANMT_DEVICE) #, cache_dir=model_path)
            mt_tokenizers[key] = tokenizer
            mt_models[key] = model
            print(f"Successfully loaded MT model: {model_name}")
        except Exception as e:
            print(f"Warning: Could not load MT model {model_name} for pair {key}: {e}")
            print("Ensure 'sentencepiece' is installed: pip install sentencepiece")
    print("MT model loading process complete.")
    return mt_tokenizers, mt_models

# --- VAD Model Loading (Silero VAD) ---
def dummy_get_speech_timestamps(audio, model, **kwargs):
    print("Warning: Using DUMMY get_speech_timestamps. VAD will not function correctly.")
    if isinstance(audio, torch.Tensor) and audio.numel() > 0 and audio.abs().sum() > 0:
        audio_len = audio.shape[-1]
        start_sample = min(int(0.1 * audio_len), audio_len - 1 if audio_len > 0 else 0)
        end_sample = min(int(0.9 * audio_len), audio_len - 1 if audio_len > 0 else 0)
        if start_sample < end_sample:
            return [{'start': start_sample, 'end': end_sample}]
    return []

def load_vad_model(config): # config might primarily be used for VAD_SILERO_SAMPLE_RATE consistency
    print("Loading Silero VAD model and utilities via torch.hub...")
    try:
        # This directly loads the model and a tuple of utility functions
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad', # Standard model, expects 16kHz
                                      force_reload=False, # Set to True to force update, False after first download
                                      onnx=False)     # Set to True for ONNX version if preferred

        if hasattr(model, 'eval'):
            model.eval()

        vad_utilities = {
            "get_speech_timestamps": utils[0],
            "save_audio": utils[1],
            "read_audio": utils[2],
            "VADIterator": utils[3],
            "collect_chunks": utils[4]
        }

        if not callable(vad_utilities["get_speech_timestamps"]):
            raise RuntimeError("Loaded 'get_speech_timestamps' from Silero VAD is not callable.")

        print("Silero VAD model and utilities loaded successfully via torch.hub.")
        return model, vad_utilities
    except Exception as e:
        print(f"ERROR: Failed to load Silero VAD model or its utilities via torch.hub: {e}")
        print("VAD will use dummy functions. Check internet connection or Silero VAD repository status.")
        return None, {"get_speech_timestamps": dummy_get_speech_timestamps}