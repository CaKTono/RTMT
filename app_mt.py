import gradio as gr
from transformers import pipeline as hf_pipeline # Alias to avoid conflict
from faster_whisper import WhisperModel
import numpy as np
import torch
import os # For MarianMT model caching if needed, though Transformers handles it

# --- Configuration ---
# For faster-whisper: "tiny", "base", "small", "medium", "large-v2", "large-v3"
# Multilingual models (not ".en" variants) are needed for language detection.
FASTER_WHISPER_MODEL_NAME = "large-v3"
ASR_MODEL = None # This will be the faster-whisper model instance

# For MarianMT translation - cache for loaded pipelines
TRANSLATION_PIPELINES_CACHE = {}

# Define some target languages for the dropdown
# Format: "Display Name": "language_code" (Helsinki-NLP Opus MT codes)
TARGET_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh", # MarianMT often uses 'zh' for general Chinese
    "Japanese": "jap",
    "Russian": "ru",
    "Arabic": "ar",
    "Indonesian": "id"
}

# --- 1. Load ASR Model (Faster-Whisper) ---
def load_asr_model_faster_whisper():
    global ASR_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For "large-v3" on CPU, float32 or int8 might be better.
    # float16 is generally for CUDA. Faster-whisper handles "auto" or "default" well.
    compute_type = "float16" if torch.cuda.is_available() else "int8" # Or "auto"
    print(f"Loading faster-whisper model: {FASTER_WHISPER_MODEL_NAME} on device: {device} with compute_type: {compute_type}")
    try:
        ASR_MODEL = WhisperModel(FASTER_WHISPER_MODEL_NAME, device=device, compute_type=compute_type)
        print(f"Faster-whisper model '{FASTER_WHISPER_MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading faster-whisper model: {e}")
        ASR_MODEL = None

# --- 2. Transcription and Language Detection Function (using Faster-Whisper) ---
def transcribe_and_detect_lang(audio_filepath):
    if ASR_MODEL is None:
        return "ASR model not loaded.", None
    if audio_filepath is None:
        return "No audio input received.", None

    try:
        print(f"Transcribing with faster-whisper: {audio_filepath}")
        segments, info = ASR_MODEL.transcribe(audio_filepath, beam_size=5)

        detected_lang = info.language
        detected_lang_confidence = info.language_probability
        print(f"Detected language: {detected_lang} with confidence: {detected_lang_confidence:.2f}")

        transcribed_text = "".join(segment.text for segment in segments).strip()
        print(f"Transcription: {transcribed_text}")
        return transcribed_text, detected_lang
    except Exception as e:
        print(f"Error during transcription with faster-whisper: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None

# --- 3. Translation Function (using MarianMT) ---
def _load_translation_pipeline(model_name):
    """Helper function to load or get a cached translation pipeline."""
    if model_name in TRANSLATION_PIPELINES_CACHE:
        print(f"Using cached translation model: {model_name}")
        return TRANSLATION_PIPELINES_CACHE[model_name]
    else:
        print(f"Loading translation model: {model_name}...")
        device_num = 0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU
        try:
            pipeline = hf_pipeline("translation", model=model_name, device=device_num)
            TRANSLATION_PIPELINES_CACHE[model_name] = pipeline
            print(f"Translation model {model_name} loaded successfully.")
            return pipeline
        except Exception as e:
            print(f"Failed to load translation model {model_name}: {e}")
            # More specific check for common Hugging Face error types
            if "is not a valid model identifier" in str(e) or "NotFoundError" in str(e) or "Connection error" in str(e):
                 # Don't cache if it's a known "not found" or connection issue, so it can be retried if network is back
                if model_name in TRANSLATION_PIPELINES_CACHE:
                    del TRANSLATION_PIPELINES_CACHE[model_name] # remove placeholder if any
            else: # Cache other exceptions to prevent repeated load attempts for truly broken models
                 TRANSLATION_PIPELINES_CACHE[model_name] = None # Mark as failed to load
            raise # Re-raise the exception to be caught by the caller

def translate_text_marian(text_to_translate, source_lang_code, target_lang_code):
    if not text_to_translate or not source_lang_code or not target_lang_code:
        return "Missing text, source language, or target language for translation."

    if source_lang_code == target_lang_code:
        return text_to_translate

    # Normalize 'zh' from faster-whisper if it's more specific like 'zh-CN'
    # For MarianMT, 'zh' is often the common code for opus-mt-zh-xx or opus-mt-xx-zh
    # Whisper typically gives 'zh' for Mandarin.
    if source_lang_code.startswith("zh-"): source_lang_code = "zh"
    if target_lang_code.startswith("zh-"): target_lang_code = "zh"

    direct_model_name = f"Helsinki-NLP/opus-mt-{source_lang_code}-{target_lang_code}"
    print(f"Attempting direct translation with: {direct_model_name}")

    try:
        translator = _load_translation_pipeline(direct_model_name)
        if translator is None: # Should be caught by _load_translation_pipeline's raise
            raise Exception(f"Model {direct_model_name} could not be loaded (returned None).")
        translated_chunks = translator(text_to_translate)
        translated_text = translated_chunks[0]['translation_text']
        print(f"Translated '{text_to_translate}' ({source_lang_code}) to '{translated_text}' ({target_lang_code}) directly.")
        return translated_text
    except Exception as e_direct:
        print(f"Direct translation model {direct_model_name} failed: {e_direct}")
        print(f"Attempting pivot translation via English: {source_lang_code} -> en -> {target_lang_code}")

        # --- Pivot Translation: Source -> English -> Target ---
        if source_lang_code == "en":
            # If source is already English, and direct failed, no pivot needed, the direct error is the final one.
            return f"Direct translation from English ({direct_model_name}) failed. No pivot possible. (Details: {str(e_direct)})"
        if target_lang_code == "en":
            # If target is English, and direct failed, this means source_lang_code -> en failed.
            return f"Direct translation to English ({direct_model_name}) failed. (Details: {str(e_direct)})"

        # 1. Translate Source to English
        text_in_english = None
        model_src_to_en_name = f"Helsinki-NLP/opus-mt-{source_lang_code}-en"
        try:
            print(f"Pivoting (1/2): Translating {source_lang_code} to en using {model_src_to_en_name}")
            translator_src_to_en = _load_translation_pipeline(model_src_to_en_name)
            if translator_src_to_en is None:
                 raise Exception(f"Pivot model {model_src_to_en_name} could not be loaded.")
            translated_chunks_en = translator_src_to_en(text_to_translate)
            text_in_english = translated_chunks_en[0]['translation_text']
            print(f"Pivoting (1/2): Translated '{text_to_translate}' ({source_lang_code}) to '{text_in_english}' (en)")
        except Exception as e_src_to_en:
            print(f"Pivoting (1/2) failed: Could not translate {source_lang_code} to en. Error: {e_src_to_en}")
            return f"Translation failed. Could not convert {source_lang_code} to English for pivoting. (Model: {model_src_to_en_name}, Error: {e_src_to_en})"

        if not text_in_english: # Should not happen if exception wasn't raised, but good check
            return f"Translation failed: Conversion from {source_lang_code} to English yielded empty text."

        # 2. Translate English to Target
        model_en_to_tgt_name = f"Helsinki-NLP/opus-mt-en-{target_lang_code}"
        try:
            print(f"Pivoting (2/2): Translating en to {target_lang_code} using {model_en_to_tgt_name}")
            translator_en_to_tgt = _load_translation_pipeline(model_en_to_tgt_name)
            if translator_en_to_tgt is None:
                raise Exception(f"Pivot model {model_en_to_tgt_name} could not be loaded.")
            translated_chunks_final = translator_en_to_tgt(text_in_english)
            final_translated_text = translated_chunks_final[0]['translation_text']
            print(f"Pivoting (2/2): Translated '{text_in_english}' (en) to '{final_translated_text}' ({target_lang_code})")
            return final_translated_text
        except Exception as e_en_to_tgt:
            print(f"Pivoting (2/2) failed: Could not translate en to {target_lang_code}. Error: {e_en_to_tgt}")
            return f"Translation failed. Converted to English ('{text_in_english}'), but could not convert English to {target_lang_code}. (Model: {model_en_to_tgt_name}, Error: {e_en_to_tgt})"

# --- 4. Main Processing Function for Gradio ---
def process_audio_and_translate(audio_filepath, target_language_display_name):
    if audio_filepath is None:
        return "Please record audio first.", "-", "No audio provided."

    transcribed_text, detected_lang_code = transcribe_and_detect_lang(audio_filepath)

    if detected_lang_code is None or transcribed_text.startswith("Error:") or not transcribed_text:
        return transcribed_text or "Transcription failed or produced empty text.", detected_lang_code or "-", "Transcription failed or empty."

    target_lang_code = TARGET_LANGUAGES.get(target_language_display_name)

    if not target_lang_code:
        return transcribed_text, f"{detected_lang_code} (Unknown target lang: {target_language_display_name})", "Invalid target language selected."

    # Find full display name for detected language, if available, otherwise use code
    detected_lang_display = detected_lang_code
    for name, code in TARGET_LANGUAGES.items():
        if code == detected_lang_code:
            detected_lang_display = f"{name} ({code})"
            break
    else: # if no display name found in our list
        detected_lang_display = f"Unknown ({detected_lang_code})"


    translated_text = translate_text_marian(transcribed_text, detected_lang_code, target_lang_code)

    return transcribed_text, detected_lang_display, translated_text

# --- 5. Gradio Interface ---
def speech_to_text_and_translation_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # Speech-to-Text and Translation App
            1. Select your desired **Target Translation Language**.
            2. Record your message using the microphone.
            3. Click "Stop recording".
            4. The transcription, detected source language, and translation will appear.
            (ASR: `faster-whisper {FASTER_WHISPER_MODEL_NAME}`, Translation: MarianMT)
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
                choices=list(TARGET_LANGUAGES.keys()), # Use display names
                value="English" # Default target language
            )

        with gr.Row():
            transcription_output_textbox = gr.Textbox(label="Transcription (Source)", lines=3, interactive=False)
            detected_lang_textbox = gr.Textbox(label="Detected Source Language", lines=1, interactive=False)

        translation_output_textbox = gr.Textbox(label="Translation (Target)", lines=3, interactive=False)

        audio_input_mic.change(
            fn=process_audio_and_translate,
            inputs=[audio_input_mic, target_lang_dropdown],
            outputs=[transcription_output_textbox, detected_lang_textbox, translation_output_textbox],
            show_progress="minimal"
        )

        gr.Markdown("---")
        gr.Markdown("Powered by Gradio, Faster-Whisper, and Hugging Face Transformers (MarianMT).")

    return demo

# --- Run the App ---
if __name__ == "__main__":
    load_asr_model_faster_whisper() # Load the faster-whisper ASR model

    if ASR_MODEL is None: # Check if faster-whisper model loaded
        print("Halting app launch as ASR model (faster-whisper) failed to load.")
    else:
        app = speech_to_text_and_translation_app()
        print("Launching Speech-to-Text and Translation Gradio app...")
        # Set share=False for local testing unless you specifically need a public link
        app.launch(debug=True, share=False)