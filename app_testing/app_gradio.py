# app_gradio.py
import gradio as gr
import numpy as np
import torch
import librosa # For resampling
import time

import config # Your configuration file
from model_loader import load_asr_model, load_mt_models, load_vad_model

# --- 1. Load Models Globally ---
print("Application starting: Loading all models...")
try:
    ASR_MODEL = load_asr_model(config)
    MT_TOKENIZERS, MT_MODELS = load_mt_models(config)
    VAD_MODEL, VAD_UTILS = load_vad_model(config)
    print("All models loaded (or attempted with fallbacks).")
except Exception as e:
    print(f"FATAL ERROR during model loading: {e}")
    print("Application cannot start. Please check model paths, configurations, and dependencies.")
    ASR_MODEL, MT_TOKENIZERS, MT_MODELS, VAD_MODEL, VAD_UTILS = None, {}, {}, None, {"get_speech_timestamps": lambda *args, **kwargs: []} # Ensure fallbacks

# --- Helper Functions ---
def resample_audio(audio_data, original_sr, target_sr):
    if original_sr == target_sr:
        return audio_data
    try:
        return librosa.resample(audio_data.astype(np.float32), orig_sr=original_sr, target_sr=target_sr)
    except Exception as e:
        print(f"Error during resampling: {e}")
        return audio_data # Return original if resampling fails

def translate_text_segment(text_to_translate, source_lang_code, target_lang_code, tokenizer, model, device):
    if not text_to_translate or source_lang_code == target_lang_code:
        return text_to_translate
    try:
        inputs = tokenizer(text_to_translate, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated_tokens = model.generate(**inputs, num_beams=config.WHISPER_BEAM_SIZE, max_length=512) # MarianMT uses num_beams too
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"Error during translation from {source_lang_code} to {target_lang_code}: {e}")
        return f"[Translation Error for '{text_to_translate}']"

# --- Constants for Audio Processing ---
AUDIO_BUFFER_MAX_DURATION_S = 15  # Max duration of audio in buffer before forcing VAD check
VAD_CHECK_INTERVAL_S = 0.3        # How often to check VAD on the buffer (seconds)
MIN_CHUNK_DURATION_S = 0.2        # Minimum audio chunk duration to send to ASR (seconds)

# Using a class to manage state per user session if Gradio reuses global scope for multiple users
# For basic demo, global is okay, but class is cleaner for state like last_vad_check_time
class AudioProcessorState:
    def __init__(self):
        self.last_vad_check_time = time.time()
        self.audio_buffer_np = np.array([], dtype=np.float32) # Internal buffer for combining chunks
        self.processing_active = False # To prevent re-entrant processing

# processor_state = AudioProcessorState() # One global state for simplicity in this example
# For true multi-user Gradio apps, state needs to be handled carefully, often via gr.State

def process_audio_stream(
    audio_chunk_with_sr,      # Tuple: (sample_rate, numpy_array_data) from gr.Audio
    target_lang_code_selected,# String: e.g., "zh" from dropdown
    # --- Gradio State Variables ---
    # These are passed in and out by Gradio to maintain state across calls FOR THIS USER'S SESSION
    # (Re-initializing them here for clarity on what they should be)
    # These are the INITIAL values if the gr.State was just gr.State()
    # If gr.State([]), initial is [], if gr.State(False), initial is False
    # They will be updated by what this function returns for these state variables
    audio_buffer_state_list,  # List of raw audio chunks (np.array), not directly used if we manage a single buffer
    full_transcript_history,  # List of strings (transcript segments)
    full_translation_history, # List of strings (translation segments)
    current_vad_speech_active_state, # Boolean: if VAD thinks speech is ongoing in current buffer segment
    last_vad_process_time_state # Timestamp of last VAD processing pass
):
    if audio_chunk_with_sr is None: # Stream just started or no audio
        return ("\n".join(full_transcript_history), "\n".join(full_translation_history),
                audio_buffer_state_list, full_transcript_history, full_translation_history,
                current_vad_speech_active_state, last_vad_process_time_state)

    sample_rate, audio_chunk_s16 = audio_chunk_with_sr
    # Convert s16 to float32, normalize
    audio_chunk_f32 = audio_chunk_s16.astype(np.float32) / np.iinfo(np.int16).max

    # Ensure mono and resample to VAD target rate (e.g., 16kHz for Silero)
    if audio_chunk_f32.ndim > 1 and audio_chunk_f32.shape[1] == 2: # Stereo
        audio_chunk_f32 = np.mean(audio_chunk_f32, axis=1)
    
    # Resample to the rate VAD expects (e.g., 16kHz)
    resampled_chunk_f32 = resample_audio(audio_chunk_f32, sample_rate, config.VAD_SILERO_SAMPLE_RATE)

    # --- Manage internal continuous audio buffer ---
    # Instead of list of chunks, concatenate into a single numpy array for VAD processing
    # For this, we'll use a temporary buffer within this function call, and rely on Gradio's state for overall history.
    # The `audio_buffer_state_list` is more for Gradio's mechanism if we were to store raw chunks.
    # Let's assume `audio_buffer_state_list` is actually our main numpy buffer passed as state.
    # This requires `gr.State(np.array([], dtype=np.float32))`
    
    # If audio_buffer_state_list is used as the actual numerical buffer:
    if not isinstance(audio_buffer_state_list, np.ndarray): # Initialize if not numpy array
        current_audio_buffer_np = np.array([], dtype=np.float32)
    else:
        current_audio_buffer_np = audio_buffer_state_list
    
    current_audio_buffer_np = np.concatenate([current_audio_buffer_np, resampled_chunk_f32])

    # Limit buffer size
    max_buffer_samples = int(AUDIO_BUFFER_MAX_DURATION_S * config.VAD_SILERO_SAMPLE_RATE)
    if len(current_audio_buffer_np) > max_buffer_samples:
        current_audio_buffer_np = current_audio_buffer_np[-max_buffer_samples:]

    speech_segments_to_process_asr = [] # List of numpy arrays (audio chunks for ASR)
    processed_upto_sample_idx = 0 # To track how much of current_audio_buffer_np is processed

    # --- VAD Logic ---
    # Only run VAD periodically or if specific conditions met
    current_time = time.time()
    time_since_last_vad_check = current_time - last_vad_process_time_state
    
    ready_for_vad = time_since_last_vad_check > VAD_CHECK_INTERVAL_S

    if VAD_MODEL and VAD_UTILS and callable(VAD_UTILS["get_speech_timestamps"]) and ready_for_vad and len(current_audio_buffer_np) > 0:
        last_vad_process_time_state = current_time # Update time of this VAD pass
        audio_tensor_for_vad = torch.from_numpy(current_audio_buffer_np).float() # VAD expects tensor

        try:
            speech_timestamps = VAD_UTILS["get_speech_timestamps"](
                audio_tensor_for_vad, VAD_MODEL,
                threshold=config.VAD_THRESHOLD,
                sampling_rate=config.VAD_SILERO_SAMPLE_RATE,
                min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
                speech_pad_ms=config.VAD_SPEECH_PAD_MS,
                min_speech_duration_ms=int(MIN_CHUNK_DURATION_S * 1000 * 0.5), # Detect short speech
                # window_size_samples=config.VAD_WINDOW_SIZE_SAMPLES, # Already a param in Silero utils
            )

            if speech_timestamps:
                current_vad_speech_active_state = 1 # Speech detected in buffer
                for ts in speech_timestamps:
                    start_sample, end_sample = ts['start'], ts['end']
                    speech_chunk_for_asr = current_audio_buffer_np[start_sample:end_sample]
                    
                    # Resample for ASR model if its target rate is different from VAD's
                    if config.VAD_SILERO_SAMPLE_RATE != config.TARGET_SAMPLE_RATE:
                        speech_chunk_for_asr = resample_audio(speech_chunk_for_asr, config.VAD_SILERO_SAMPLE_RATE, config.TARGET_SAMPLE_RATE)

                    if len(speech_chunk_for_asr) / config.TARGET_SAMPLE_RATE >= MIN_CHUNK_DURATION_S:
                        speech_segments_to_process_asr.append(speech_chunk_for_asr)
                    
                    processed_upto_sample_idx = max(processed_upto_sample_idx, end_sample)
            else: # No speech detected by VAD in current buffer
                # If speech was active and now it's not, implies end of utterance.
                # However, relying on min_silence_duration_ms is often better.
                current_vad_speech_active_state = 0 
        
        except Exception as e:
            print(f"VAD Error: {e}")
            # Fallback: if VAD fails, maybe process long buffer (less ideal)
            if len(current_audio_buffer_np) > (config.TARGET_SAMPLE_RATE * 3): # e.g. 3 seconds
                chunk_for_asr = resample_audio(current_audio_buffer_np, config.VAD_SILERO_SAMPLE_RATE, config.TARGET_SAMPLE_RATE)
                speech_segments_to_process_asr.append(chunk_for_asr)
                processed_upto_sample_idx = len(current_audio_buffer_np)

    # If no VAD, or VAD didn't find anything conclusive, but buffer is long
    if not speech_segments_to_process_asr and len(current_audio_buffer_np) > (config.TARGET_SAMPLE_RATE * 5) and not VAD_MODEL :
        print("No VAD or VAD inconclusive, processing long buffer as a fallback.")
        chunk_for_asr = resample_audio(current_audio_buffer_np, config.VAD_SILERO_SAMPLE_RATE, config.TARGET_SAMPLE_RATE)
        speech_segments_to_process_asr.append(chunk_for_asr)
        processed_upto_sample_idx = len(current_audio_buffer_np)

    # Update main audio buffer by removing processed part
    if processed_upto_sample_idx > 0:
        current_audio_buffer_np = current_audio_buffer_np[processed_upto_sample_idx:]

    # --- ASR & MT for processed speech segments ---
    if ASR_MODEL and speech_segments_to_process_asr:
        for speech_chunk_asr_final in speech_segments_to_process_asr:
            try:
                segments_iter, info = ASR_MODEL.transcribe(
                    speech_chunk_asr_final,
                    beam_size=config.WHISPER_BEAM_SIZE,
                    language=None, # Auto-detect language
                    vad_filter=False, # We are doing our own VAD
                )
                detected_lang_code = info.language
                
                full_text_from_segments = []
                for segment in segments_iter:
                    segment_text = segment.text.strip()
                    if segment_text:
                        full_text_from_segments.append(segment_text)
                
                if full_text_from_segments:
                    final_transcript_text = " ".join(full_text_from_segments)
                    transcript_entry = f"[{detected_lang_code.upper()}]: {final_transcript_text}"
                    full_transcript_history.append(transcript_entry)

                    # MT
                    translated_text_for_display = ""
                    if target_lang_code_selected and detected_lang_code != target_lang_code_selected:
                        model_key = f"{detected_lang_code}-{target_lang_code_selected}"
                        if model_key in MT_MODELS and model_key in MT_TOKENIZERS:
                            translated_text_for_display = translate_text_segment(
                                final_transcript_text, detected_lang_code, target_lang_code_selected,
                                MT_TOKENIZERS[model_key], MT_MODELS[model_key], config.MARIANMT_DEVICE
                            )
                            translation_entry = f"[{target_lang_code_selected.upper()}]: {translated_text_for_display}"
                        else:
                            translation_entry = f"[No model for {model_key}]"
                    else: # Same language or no specific target selected for translation
                        translation_entry = f"[{detected_lang_code.upper()}]: {final_transcript_text}" # Or simply "" if no translation needed
                    full_translation_history.append(translation_entry)

            except Exception as e:
                print(f"ASR/MT Error for a chunk: {e}")
                full_transcript_history.append("[ASR Error]")
    
    # Cap history length
    MAX_HISTORY_ITEMS = 20
    full_transcript_history = full_transcript_history[-MAX_HISTORY_ITEMS:]
    full_translation_history = full_translation_history[-MAX_HISTORY_ITEMS:]

    return ("\n".join(full_transcript_history),
            "\n".join(full_translation_history),
            current_audio_buffer_np, # Return the updated numpy buffer
            full_transcript_history,
            full_translation_history,
            current_vad_speech_active_state,
            last_vad_process_time_state)


def clear_history_fn():
    # Returns initial values for all state variables and output components
    # return "", "", np.array([], dtype=np.float32), [], [], 0, time.time()
    return "", "", None, [], [], 0, time.time()

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Real-time Speech Translation Engine")
    gr.Markdown("Select target language, then click 'Record from microphone'. Speak clearly into your microphone.")

    # --- State Variables for the User Session ---
    # These hold session-specific data that persist across streaming calls.
    # Initialize with appropriate empty/default values.
    # Using a numpy array for the audio buffer state is more efficient for concatenation.
    # audio_buffer_state = gr.State(np.array([], dtype=np.float32))
    audio_buffer_state = gr.State(None) # Initialize with None
    transcript_history_state = gr.State([]) # List of transcript strings
    translation_history_state = gr.State([])# List of translation strings
    vad_speech_active_gr_state = gr.State(0) # Integer: 0 for False, 1 for True
    last_vad_time_gr_state = gr.State(time.time()) # Timestamp of the last VAD processing pass

    with gr.Row():
        with gr.Column(scale=1):
            target_language_dd = gr.Dropdown(
                label="Translate Speech To",
                choices=[(name, code) for code, name in config.SUPPORTED_LANGUAGES.items()],
                value="zh", # Default to Chinese
                interactive=True
            )
            audio_input_mic = gr.Audio(
                sources=["microphone"],
                type="numpy", # Returns (sample_rate, numpy_array_data)
                streaming=True, # Crucial for real-time
                label="Live Microphone Input",
                # waveform_options=gr.WaveformOptions(show_controls=True) # Optional: more waveform controls
            )
            clear_btn = gr.Button("Clear History & Reset Audio")

        with gr.Column(scale=2):
            transcript_output_tb = gr.Textbox(
                label="Live Transcript", lines=10, interactive=False, autoscroll=True
            )
            translation_output_tb = gr.Textbox(
                label="Live Translation", lines=10, interactive=False, autoscroll=True
            )
    
    gr.Markdown("--- \n *Logs and Status (for debugging):*")
    status_html = gr.HTML("Status: Initializing...") # For simple status updates if needed

    # Connect the streaming audio input to the processing function
    audio_input_mic.stream(
        fn=process_audio_stream,
        inputs=[
            audio_input_mic,
            target_language_dd,
            audio_buffer_state,
            transcript_history_state,
            translation_history_state,
            vad_speech_active_gr_state,
            last_vad_time_gr_state
        ],
        outputs=[
            transcript_output_tb,
            translation_output_tb,
            audio_buffer_state, # Output state to update it
            transcript_history_state,
            translation_history_state,
            vad_speech_active_gr_state,
            last_vad_time_gr_state
        ],
        # show_progress="hidden" # Can be "full", "minimal", "hidden"
        api_name="speech_stream" # For API access if enabled
    )

    clear_btn.click(
        fn=clear_history_fn,
        inputs=None, # No direct inputs from UI needed for clear
        outputs=[
            transcript_output_tb,
            translation_output_tb,
            audio_buffer_state,
            transcript_history_state,
            translation_history_state,
            vad_speech_active_gr_state,
            last_vad_time_gr_state
            # Potentially also audio_input_mic to clear its display, though usually not needed
        ],
        api_name="clear_history"
    )

if __name__ == "__main__":
    print("Attempting to launch Gradio demo...")
    # Using demo.queue() is important for handling multiple requests if streaming or long tasks.
    # Set share=True to create a public link (requires internet & can be slow from China).
    # Set share=False for local access only (http://127.0.0.1:7860 or http://0.0.0.0:7860).
    try:
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
    except Exception as e:
        print(f"Error launching Gradio demo: {e}")
        print("This might be due to the 'TypeError: argument of type 'bool' is not iterable' if Gradio version is old or has a conflict.")
        print("Ensure you have run 'pip install --upgrade gradio'.")
        print("If the problem persists, it might be a deeper issue with Gradio's state handling or API generation in your environment.")