import gradio as gr
from transformers import pipeline as hf_pipeline # Alias to avoid conflict
from faster_whisper import WhisperModel
import numpy as np
import torch
import os

# --- Configuration ---
FASTER_WHISPER_MODEL_NAME = "large-v3"
ASR_MODEL = None

# --- NLLB Configuration (Translation Specialist) ---
# Using Meta's NLLB, a model purpose-built for high-quality translation.
# This is the robust solution to the problems we've seen.
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
TRANSLATION_PIPELINE = None

# NLLB uses different language codes than Whisper. We must map them.
# Whisper Code -> NLLB Code
WHISPER_TO_NLLB_MAP = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "ar": "ara_Arab",
    "id": "ind_Latn"
}

# The user-facing dropdown can still use friendly names and simple codes
TARGET_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Russian": "ru",
    "Arabic": "ar",
    "Indonesian": "id"
}


# --- 1. Load ASR Model (Faster-Whisper) ---
def load_asr_model_faster_whisper():
    """Loads the Faster-Whisper ASR model."""
    global ASR_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    print(f"Loading faster-whisper model: {FASTER_WHISPER_MODEL_NAME} on device: {device} with compute_type: {compute_type}")
    try:
        ASR_MODEL = WhisperModel(FASTER_WHISPER_MODEL_NAME, device=device, compute_type=compute_type)
        print(f"Faster-whisper model '{FASTER_WHISPER_MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading faster-whisper model: {e}")
        ASR_MODEL = None

# --- 2. Load NLLB Translation Model ---
def load_translation_model():
    """Loads the NLLB translation model."""
    global TRANSLATION_PIPELINE
    if TRANSLATION_PIPELINE is not None:
        print("NLLB Translation model already loaded.")
        return

    device_num = 0 if torch.cuda.is_available() else -1
    print(f"Loading NLLB translation model: {TRANSLATION_MODEL_NAME}...")
    try:
        # For NLLB, the task is "translation" and we specify languages during the call
        TRANSLATION_PIPELINE = hf_pipeline("translation", model=TRANSLATION_MODEL_NAME, device=device_num)
        print(f"NLLB Translation model '{TRANSLATION_MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading NLLB translation model: {e}")
        TRANSLATION_PIPELINE = None


# --- 3. Transcription and Language Detection Function (using Faster-Whisper) ---
def transcribe_and_detect_lang(audio_filepath):
    """Transcribes audio and detects the source language."""
    if ASR_MODEL is None: return "ASR model not loaded.", None
    if audio_filepath is None: return "No audio input received.", None
    try:
        segments, info = ASR_MODEL.transcribe(audio_filepath, beam_size=5)
        detected_lang_code = info.language
        transcribed_text = "".join(segment.text for segment in segments).strip()
        print(f"Transcription: {transcribed_text} (Detected Lang Code: {detected_lang_code})")
        return transcribed_text, detected_lang_code
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error: {str(e)}", None

# --- 4. Translation Function (using NLLB) ---
def translate_text_nllb(text_to_translate, source_whisper_code, target_whisper_code):
    """Translates text using the loaded NLLB model."""
    if TRANSLATION_PIPELINE is None: return "NLLB Translation model not loaded."
    if not text_to_translate: return "" # Return empty if transcription is empty
    if source_whisper_code == target_whisper_code: return text_to_translate

    # Map the simple whisper codes to the required NLLB codes
    src_lang = WHISPER_TO_NLLB_MAP.get(source_whisper_code)
    tgt_lang = WHISPER_TO_NLLB_MAP.get(target_whisper_code)

    if not src_lang:
        return f"ERROR: The detected source language '{source_whisper_code}' is not supported for translation."
    if not tgt_lang:
        return f"ERROR: The target language '{target_whisper_code}' is not supported for translation."

    print(f"Translating with NLLB from {src_lang} to {tgt_lang}...")
    try:
        # NLLB pipeline requires src_lang and tgt_lang parameters
        result = TRANSLATION_PIPELINE(text_to_translate, src_lang=src_lang, tgt_lang=tgt_lang)
        translated_text = result[0]['translation_text']
        print(f"NLLB Translated '{text_to_translate}' to '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"Error during NLLB translation: {e}")
        return f"Error during translation: {str(e)}"

# --- 5. Main Processing Function for Gradio ---
def process_audio_and_translate(audio_filepath, target_language_display_name):
    """Orchestrates the ASR and Translation process."""
    if audio_filepath is None: return "Please record audio first.", "-", "No audio provided."

    transcribed_text, detected_lang_code = transcribe_and_detect_lang(audio_filepath)

    if detected_lang_code is None or transcribed_text.startswith("Error:"):
        return transcribed_text or "Transcription failed.", detected_lang_code or "-", "Translation skipped."

    # Get the simple 'en', 'id', etc. code for the target language from the dropdown
    target_lang_code = TARGET_LANGUAGES.get(target_language_display_name)

    # Get a display name for the detected language for the UI
    detected_lang_display = f"Unknown ({detected_lang_code})"
    for name, code in TARGET_LANGUAGES.items():
        if code == detected_lang_code:
            detected_lang_display = f"{name} ({code})"
            break

    translated_text = translate_text_nllb(transcribed_text, detected_lang_code, target_lang_code)

    return transcribed_text, detected_lang_display, translated_text

# --- 6. Gradio Interface ---
def speech_to_text_and_translation_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
            # Speech-to-Text and Translation (NLLB Specialist Version)
            This app uses **Meta's NLLB model**, a specialist in translation, for more accurate and reliable results.
            (ASR: `faster-whisper/{FASTER_WHISPER_MODEL_NAME}`, Translation: `{TRANSLATION_MODEL_NAME}`)
            """)
        with gr.Row():
            audio_input_mic = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio")
        with gr.Row():
            target_lang_dropdown = gr.Dropdown(label="Target Translation Language", choices=list(TARGET_LANGUAGES.keys()), value="English")
        with gr.Row():
            transcription_output_textbox = gr.Textbox(label="Transcription (Source)", lines=3, interactive=False)
            detected_lang_textbox = gr.Textbox(label="Detected Source Language", lines=1, interactive=False)
        translation_output_textbox = gr.Textbox(label="Translation (Target)", lines=3, interactive=False)
        audio_input_mic.change(fn=process_audio_and_translate, inputs=[audio_input_mic, target_lang_dropdown], outputs=[transcription_output_textbox, detected_lang_textbox, translation_output_textbox], show_progress="minimal")
        gr.Markdown("---")
        gr.Markdown("Powered by Gradio, Faster-Whisper, and Hugging Face Transformers (Meta's NLLB).")

    return demo

# --- Run the App ---
if __name__ == "__main__":
    print("Starting application setup...")
    load_asr_model_faster_whisper()
    load_translation_model()

    if ASR_MODEL is None or TRANSLATION_PIPELINE is None:
        print("Halting app launch as a required model failed to load. Please check the console for errors.")
    else:
        app = speech_to_text_and_translation_app()
        print("Launching Speech-to-Text and Translation Gradio app...")
        app.launch(debug=True, share=True)