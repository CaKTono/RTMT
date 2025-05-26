# config.py

# --- ASR (Faster-Whisper) Configurations ---
WHISPER_MODEL_SIZE = "large-v3" # "tiny", "base", "small", "medium", "large-v2", "large-v3"
                               # "large-v3" is most accurate but very resource-intensive.
                               # "medium" or "small" are good balances. "base" for speed.
WHISPER_DEVICE = "cuda"       # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16" # "float16" (GPU), "int8" (GPU), "float32" (CPU)
WHISPER_BEAM_SIZE = 5

# --- MT (MarianMT) Configurations ---
MARIANMT_DEVICE = "cuda"      # "cuda" or "cpu"

# --- Audio Processing Configurations ---
TARGET_SAMPLE_RATE = 16000    # Whisper standard
VAD_SILERO_SAMPLE_RATE = 16000 # Silero VAD works best at 16kHz or 8kHz

# --- Supported Languages ---
# Add language codes and names as needed. Ensure corresponding MT models exist.
SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "id": "Indonesian"
}

# --- MarianMT Model Mapping ---
# Keys are "src_lang-tgt_lang", values are Hugging Face model names.
# Ensure you have models for translations you want to support.
MARIANMT_MODEL_MAP = {
    # English to X
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "en-id": "Helsinki-NLP/opus-mt-en-id",
    "en-ja": "Helsinki-NLP/opus-mt-en-jap", # Check exact model name for Japanese
    # "en-ko": "Helsinki-NLP/opus-mt-en-ko",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",

    # Chinese to English (Example)
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",

    # Indonesian to English (Example)
    "id-en": "Helsinki-NLP/opus-mt-id-en",

    # Add other pairs as needed. For example, if you detect Spanish and want to translate to Chinese:
    # "es-zh": "Helsinki-NLP/opus-mt-es-zh", # (You'd need to find or train such a model)
}

# --- VAD (Silero VAD) Parameters ---
VAD_THRESHOLD = 0.5  # Confidence threshold for speech detection (0 to 1)
VAD_MIN_SILENCE_DURATION_MS = 250 # Minimum silence duration in ms to consider a pause
VAD_SPEECH_PAD_MS = 100 # Pad speech chunks with silence at start/end (ms)
                        # Helps avoid cutting off words.
VAD_WINDOW_SIZE_SAMPLES = 512 # Samples, affects VAD processing chunk size for Silero.
                              # Common values: 256, 512, 1024. Silero examples use 512 for 16kHz.