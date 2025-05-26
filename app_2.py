import gradio as gr
import gradio as gr
from transformers import pipeline
import numpy as np
# import soundfile as sf # Not strictly needed for this VAD flow but can be kept
import os
import time
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import torch
import torchaudio

# --- Configuration ---
ASR_MODEL = "openai/whisper-base.en"
ASR_TRANSCRIBER = None  # Renamed for clarity

# --- VAD Configuration ---
VAD_MODEL = None
VAD_UTILS = None
VAD_SAMPLE_RATE = 16000 # Silero VAD expects 16kHz
VAD_THRESHOLD = 0.4   # Speech probability threshold (0.0 to 1.0)
MIN_SILENCE_DURATION_MS = 600 # Min silence to trigger end of segment
MIN_SPEECH_DURATION_MS = 250   # Min duration of speech to process
# STREAM_CHUNK_DURATION_S = 0.2 # This was defined but not used for mic_input_stream.stream_every
                                # Gradio's default (usually 0.1s) will be used for streaming frequency.

# --- 1. Load Models (ASR and VAD) ---
def load_models():
    global ASR_TRANSCRIBER, VAD_MODEL, VAD_UTILS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        print(f"Loading ASR model: {ASR_MODEL}...")
        ASR_TRANSCRIBER = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device)
        print(f"ASR model loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading ASR model: {e}")
        ASR_TRANSCRIBER = None

    try:
        print("Loading Silero VAD model...")
        torch.set_num_threads(1) # Recommended for Silero VAD
        # VAD model will be loaded on CPU by default by torch.hub.load
        VAD_MODEL, VAD_UTILS = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False,
                                              onnx=False)
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = VAD_UTILS
        print("Silero VAD model loaded successfully (typically on CPU).")
    except Exception as e:
        print(f"Error loading Silero VAD model: {e}")
        VAD_MODEL = None

# --- 2. Transcription and Plotting Function ---
def transcribe_and_plot_audio_segment(audio_segment_data, gain_reduction_factor=1.0):
    global ASR_TRANSCRIBER
    if ASR_TRANSCRIBER is None:
        return "ASR model not loaded.", None
    if audio_segment_data is None or audio_segment_data[1].size == 0:
        return "", None

    try:
        sample_rate, data_raw = audio_segment_data # data_raw is already float32 from our buffer

        print(f"Processing segment for ASR. SR: {sample_rate}, Duration: {len(data_raw)/sample_rate:.2f}s, Gain: {gain_reduction_factor}")

        data_float_reduced = data_raw * gain_reduction_factor
        data_processed = np.clip(data_float_reduced, -1.0, 1.0)
        
        if len(data_processed.shape) > 1:
            data_processed = np.mean(data_processed, axis=1)

        transcription_result = ASR_TRANSCRIBER({"sampling_rate": sample_rate, "raw": data_processed})
        text = transcription_result.get("text", "No text in result").strip()
        print(f"  ASR Transcription: {text}")

        fig, ax = plt.subplots(figsize=(6, 2.5))
        time_axis = np.linspace(0, len(data_processed) / sample_rate, num=len(data_processed))
        ax.plot(time_axis, data_processed, lw=0.5)
        ax.set_title(f"Processed Speech Segment ({len(data_processed)/sample_rate:.2f}s)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        ax.set_ylim([-1.05, 1.05]); ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        return text, fig

    except Exception as e:
        print(f"Error during transcription/plotting segment: {e}")
        import traceback; traceback.print_exc()
        fig, ax = plt.subplots(figsize=(6, 2)); ax.text(0.5, 0.5, "Error processing segment", ha='center', va='center'); plt.tight_layout()
        return f"Error: {str(e)}", fig

# --- 3. Gradio Interface ---
def continuous_transcription_app_vad():
    initial_vad_speech_buffer = []
    initial_vad_current_sr = None
    initial_vad_is_speaking = False
    initial_vad_silence_duration_accumulator_ms = 0 # Renamed for clarity

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 Continuous ASR with Silero VAD (Fixed Windowing)")
        gr.Markdown(
            "Speak into your microphone. Audio is segmented by VAD before transcription. "
            "Adjust gain if needed. **For persistent high gain, check OS microphone settings!**"
        )

        vad_speech_buffer_state = gr.State(initial_vad_speech_buffer)
        vad_current_sr_state = gr.State(initial_vad_current_sr)
        vad_is_speaking_state = gr.State(initial_vad_is_speaking)
        vad_silence_accumulator_state = gr.State(initial_vad_silence_duration_accumulator_ms) # Renamed
        
        full_transcript_var = gr.State("")

        with gr.Row():
            with gr.Column(scale=2):
                output_textbox = gr.Textbox(label="Live Transcription", lines=10, interactive=False, autoscroll=True)
            with gr.Column(scale=1):
                plot_output = gr.Plot(label="Last Processed Speech Segment")
                gain_slider = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                                        label="Software Gain Reduction", info="1.0 = no reduction")

        mic_input_stream = gr.Audio(
            sources=["microphone"], type="numpy", streaming=True,
            label="Microphone Input (Streaming)",
            # stream_every=0.1 # Explicitly set Gradio's chunking interval if desired (in seconds)
        )

        def process_vad_audio_stream(new_audio_chunk_tuple,
                                     current_full_transcript,
                                     gain_factor,
                                     current_speech_buffer,
                                     current_sr_for_buffer,
                                     is_currently_speaking,
                                     current_silence_ms_accumulator): # Renamed parameter
            
            text_to_display = current_full_transcript
            plot_to_display = None # Keep current plot by default
            
            updated_speech_buffer = list(current_speech_buffer) # Ensure it's a mutable list
            updated_sr_for_buffer = current_sr_for_buffer
            updated_is_speaking = is_currently_speaking
            updated_silence_ms_accumulator = current_silence_ms_accumulator # Renamed

            if new_audio_chunk_tuple is None or VAD_MODEL is None:
                return (current_full_transcript, text_to_display, plot_to_display, 
                        updated_speech_buffer, updated_sr_for_buffer, updated_is_speaking, updated_silence_ms_accumulator)

            sample_rate, audio_chunk_int16 = new_audio_chunk_tuple
            
            if np.issubdtype(audio_chunk_int16.dtype, np.integer):
                audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / np.iinfo(audio_chunk_int16.dtype).max
            elif audio_chunk_int16.dtype == np.float32:
                audio_chunk_float32 = audio_chunk_int16.copy() # Use a copy
            else: 
                audio_chunk_float32 = audio_chunk_int16.astype(np.float32) 

            # Ensure audio_chunk_float32 is 1D (mono) for resampling
            if len(audio_chunk_float32.shape) > 1:
                audio_chunk_float32_mono = np.mean(audio_chunk_float32, axis=1)
            else:
                audio_chunk_float32_mono = audio_chunk_float32

            # Resample audio chunk to VAD_SAMPLE_RATE (16kHz) for VAD decision
            if sample_rate != VAD_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=VAD_SAMPLE_RATE)
                audio_tensor_for_vad = resampler(torch.from_numpy(audio_chunk_float32_mono).unsqueeze(0)).squeeze(0)
            else:
                audio_tensor_for_vad = torch.from_numpy(audio_chunk_float32_mono)
            
            # --- VAD Processing with Correct Window Sizes (FIX APPLIED HERE) ---
            window_size_samples = 512  # Expected by Silero VAD at 16kHz
            if VAD_SAMPLE_RATE == 8000: # Safety check if VAD_SAMPLE_RATE is ever changed
                window_size_samples = 256
            
            speech_probs_for_gradio_chunk = []
            num_samples_in_resampled_chunk = audio_tensor_for_vad.shape[0]
            current_pos_vad = 0
            
            while current_pos_vad < num_samples_in_resampled_chunk:
                end_pos_vad = current_pos_vad + window_size_samples
                window_tensor = audio_tensor_for_vad[current_pos_vad:end_pos_vad]
                actual_window_len = window_tensor.shape[0]

                if actual_window_len == 0: break

                if actual_window_len < window_size_samples:
                    padding_needed = window_size_samples - actual_window_len
                    padding = torch.zeros(padding_needed, dtype=window_tensor.dtype, device=window_tensor.device)
                    window_tensor = torch.cat((window_tensor, padding))
                
                # VAD model is on CPU, ensure tensor is on CPU.
                prob = VAD_MODEL(window_tensor.cpu(), VAD_SAMPLE_RATE).item()
                speech_probs_for_gradio_chunk.append(prob)
                
                current_pos_vad += actual_window_len
            
            if not speech_probs_for_gradio_chunk:
                speech_prob = 0.0
            else:
                speech_prob = max(speech_probs_for_gradio_chunk) # Use max prob from sub-windows
            # --- End of VAD Windowing Fix ---
            
            chunk_duration_ms = (len(audio_chunk_float32_mono) / sample_rate) * 1000

            if speech_prob > VAD_THRESHOLD:
                if not updated_is_speaking:
                    print(f"VAD: Speech started (Prob: {speech_prob:.2f})")
                    updated_speech_buffer = [] 
                    updated_sr_for_buffer = sample_rate
                
                updated_is_speaking = True
                updated_speech_buffer.append(audio_chunk_float32_mono) # Buffer original SR, mono, float32 chunk
                updated_silence_ms_accumulator = 0
            
            elif updated_is_speaking: # Speech was active, now silence for this Gradio chunk
                updated_silence_ms_accumulator += chunk_duration_ms
                print(f"VAD: Potential silence after speech. Accumulator: {updated_silence_ms_accumulator:.0f}ms (Prob: {speech_prob:.2f})")
                
                if updated_silence_ms_accumulator >= MIN_SILENCE_DURATION_MS:
                    print(f"VAD: Speech ended (silence {updated_silence_ms_accumulator:.0f}ms >= {MIN_SILENCE_DURATION_MS}ms). Processing buffer.")
                    updated_is_speaking = False
                    
                    if updated_speech_buffer and updated_sr_for_buffer is not None:
                        complete_speech_segment = np.concatenate(updated_speech_buffer)
                        segment_duration_ms = (len(complete_speech_segment) / updated_sr_for_buffer) * 1000
                        
                        if segment_duration_ms >= MIN_SPEECH_DURATION_MS:
                            print(f"  Segment long enough ({segment_duration_ms:.0f}ms). Sending to ASR.")
                            segment_text, fig = transcribe_and_plot_audio_segment(
                                (updated_sr_for_buffer, complete_speech_segment), gain_factor
                            )
                            if segment_text and not segment_text.startswith("Error:"):
                                text_to_display = current_full_transcript + (" " if current_full_transcript else "") + segment_text
                            plot_to_display = fig
                        else:
                            print(f"  Segment too short ({segment_duration_ms:.0f}ms). Discarding.")
                    
                    updated_speech_buffer = [] 
                    updated_silence_ms_accumulator = 0 # Reset after processing segment
            # else: silence and no speech was active, do nothing

            return (text_to_display, text_to_display, plot_to_display, 
                    updated_speech_buffer, updated_sr_for_buffer, updated_is_speaking, updated_silence_ms_accumulator)

        mic_input_stream.stream(
            fn=process_vad_audio_stream,
            inputs=[mic_input_stream, full_transcript_var, gain_slider, 
                    vad_speech_buffer_state, vad_current_sr_state, vad_is_speaking_state, vad_silence_accumulator_state], # Renamed state
            outputs=[full_transcript_var, output_textbox, plot_output,
                     vad_speech_buffer_state, vad_current_sr_state, vad_is_speaking_state, vad_silence_accumulator_state], # Renamed state
            show_progress="hidden"
        )

        def clear_all_vad():
            fig, ax = plt.subplots(figsize=(6, 2)); ax.text(0.5, 0.5, "Cleared", ha='center', va='center'); plt.tight_layout()
            return ("", "", fig, 
                    initial_vad_speech_buffer, initial_vad_current_sr, 
                    initial_vad_is_speaking, initial_vad_silence_duration_accumulator_ms) # Renamed state

        clear_button = gr.Button("Clear Transcript & Plot")
        clear_button.click(
            fn=clear_all_vad, inputs=None,
            outputs=[full_transcript_var, output_textbox, plot_output,
                     vad_speech_buffer_state, vad_current_sr_state, vad_is_speaking_state, vad_silence_accumulator_state] # Renamed state
        )
        
        gr.Markdown("---")
        gr.Markdown("VAD segments audio based on speech detection. Adjust Gain Slider if needed. "
                    "For best results, ensure a quiet environment and clear speech.")
    return demo

# --- Run the App ---
if __name__ == "__main__":
    load_models()
    if ASR_TRANSCRIBER is None or VAD_MODEL is None:
        print("Halting app launch as one or more models failed to load.")
    else:
        app = continuous_transcription_app_vad()
        print("Launching Gradio app with VAD (fixed windowing)...")
        app.launch(debug=True, share=False)