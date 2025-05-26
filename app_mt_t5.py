import gradio as gr
from transformers import pipeline as hf_pipeline # Alias to avoid conflict
from faster_whisper import WhisperModel
import numpy as np
import torch
import os

# --- Configuration ---
# For faster-whisper: "tiny", "base", "small", "medium", "large-v2", "large-v3"
FASTER_WHISPER_MODEL_NAME = "large-v3"
ASR_MODEL = None

# --- NEW: Flan-T5 Configuration ---
# Using Flan-T5, which is instruction-tuned and much better for out-of-the-box translation.
# 'google/flan-t5-large' (~3GB VRAM) is a good balance of power and size.
# Other options: 'google/flan-t5-base', 'google/flan-t5-xl'
TRANSLATION_MODEL_NAME = "google/flan-t5-large"
TRANSLATION_PIPELINE = None # This will hold the single Flan-T5 pipeline

# Map the language codes from Whisper to the full language names T5 expects.
LANGUAGE_CODE_TO_NAME_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
    "id": "Indonesian"
}

# Define some target languages for the dropdown using the full names
TARGET_LANGUAGES = {
    "English": "English",
    "Spanish": "Spanish",
    "French": "French",
    "German": "German",
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Russian": "Russian",
    "Arabic": "Arabic",
    "Indonesian": "Indonesian"
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

# --- NEW: Load Flan-T5 Translation Model ---
def load_translation_model():
    """Loads the Flan-T5 translation model."""
    global TRANSLATION_PIPELINE
    if TRANSLATION_PIPELINE is not None:
        print("Flan-T5 Translation model already loaded.")
        return

    device_num = 0 if torch.cuda.is_available() else -1
    print(f"Loading Flan-T5 translation model: {TRANSLATION_MODEL_NAME}...")
    try:
        TRANSLATION_PIPELINE = hf_pipeline("text2text-generation", model=TRANSLATION_MODEL_NAME, device=device_num)
        print(f"Flan-T5 Translation model '{TRANSLATION_MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading Flan-T5 translation model: {e}")
        print("This may be due to insufficient VRAM/RAM or a missing model.")
        TRANSLATION_PIPELINE = None


# --- 2. Transcription and Language Detection Function (using Faster-Whisper) ---
def transcribe_and_detect_lang(audio_filepath):
    """Transcribes audio and detects the source language."""
    if ASR_MODEL is None:
        return "ASR model not loaded.", None
    if audio_filepath is None:
        return "No audio input received.", None

    try:
        print(f"Transcribing with faster-whisper: {audio_filepath}")
        segments, info = ASR_MODEL.transcribe(audio_filepath, beam_size=5)

        detected_lang_code = info.language
        print(f"Detected language code: {detected_lang_code} with confidence: {info.language_probability:.2f}")

        transcribed_text = "".join(segment.text for segment in segments).strip()
        print(f"Transcription: {transcribed_text}")
        return transcribed_text, detected_lang_code
    except Exception as e:
        print(f"Error during transcription with faster-whisper: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None

# --- 3. Translation Function (using Google Flan-T5) ---
def translate_text_flan_t5(text_to_translate, source_lang_name, target_lang_name):
    """Translates text using the loaded Google Flan-T5 model."""
    if TRANSLATION_PIPELINE is None:
        return "Flan-T5 Translation model not loaded. Please check logs for errors."
    if not text_to_translate or not source_lang_name or not target_lang_name:
        return "Missing text, source language, or target language for translation."

    if source_lang_name == target_lang_name:
        return text_to_translate

    # CORRECTED PROMPT: Ensure there are no extra quotes around the text variable.
    prompt = f"translate {source_lang_name} to {target_lang_name}: {text_to_translate}"
    print(f"Flan-T5 Prompt: \"{prompt}\"")

    try:
        translated_chunks = TRANSLATION_PIPELINE(prompt, max_length=1024)
        translated_text = translated_chunks[0]['generated_text']
        print(f"Translated '{text_to_translate}' ({source_lang_name}) to '{translated_text}' ({target_lang_name})")
        return translated_text
    except Exception as e:
        print(f"Error during Flan-T5 translation: {e}")
        return f"Error during translation: {str(e)}"

# --- 4. Main Processing Function for Gradio ---
def process_audio_and_translate(audio_filepath, target_language_name):
    """Orchestrates the ASR and Translation process."""
    if audio_filepath is None:
        return "Please record audio first.", "-", "No audio provided."

    # Step 1: Transcribe and detect language
    transcribed_text, detected_lang_code = transcribe_and_detect_lang(audio_filepath)

    if detected_lang_code is None or transcribed_text.startswith("Error:") or not transcribed_text:
        return transcribed_text or "Transcription failed.", detected_lang_code or "-", "Translation skipped."

    # Step 2: Map detected language code to full name for T5
    source_lang_name = LANGUAGE_CODE_TO_NAME_MAP.get(detected_lang_code, detected_lang_code)
    detected_lang_display = f"{source_lang_name} ({detected_lang_code})"

    # Step 3: Translate using Flan-T5
    translated_text = translate_text_flan_t5(transcribed_text, source_lang_name, target_language_name)

    return transcribed_text, detected_lang_display, translated_text

# --- 5. Gradio Interface ---
def speech_to_text_and_translation_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # Speech-to-Text and Translation App (Flan-T5 Version)
            1. Select your desired **Target Translation Language**.
            2. Record your message using the microphone.
            3. The transcription, detected source language, and translation will appear.
            (ASR: `faster-whisper/{FASTER_WHISPER_MODEL_NAME}`, Translation: `{TRANSLATION_MODEL_NAME}`)
            """
        )

        with gr.Row():
            audio_input_mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Audio"
            )

        with gr.Row():
            target_lang_dropdown = gr.Dropdown(
                label="Target Translation Language",
                choices=list(TARGET_LANGUAGES.keys()),
                value="English" # Default target
            )

        with gr.Row():
            transcription_output_textbox = gr.Textbox(label="Transcription (Source)", lines=3, interactive=False)
            detected_lang_textbox = gr.Textbox(label="Detected Source Language", lines=1, interactive=False)

        translation_output_textbox = gr.Textbox(label="Translation (Target)", lines=3, interactive=False)

        # I've updated the function call here to the new function name
        audio_input_mic.change(
            fn=process_audio_and_translate,
            inputs=[audio_input_mic, target_lang_dropdown],
            outputs=[transcription_output_textbox, detected_lang_textbox, translation_output_textbox],
            show_progress="minimal"
        )

        gr.Markdown("---")
        gr.Markdown("Powered by Gradio, Faster-Whisper, and Hugging Face Transformers (Google Flan-T5).")

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