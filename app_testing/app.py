import gradio as gr
from transformers import pipeline
import numpy as np
import soundfile as sf # Keep for potential future use, not strictly for this flow
import os
import time # For unique filenames if saving plots, or other time-related ops
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt

# --- Configuration ---
ASR_MODEL = "openai/whisper-base.en"
transcriber = None # Will be loaded in main

# --- 1. Load ASR Model ---
def load_asr_model():
    global transcriber
    try:
        print(f"Loading ASR model: {ASR_MODEL}...")
        transcriber = pipeline("automatic-speech-recognition", model=ASR_MODEL)
        print("ASR model loaded successfully.")
    except Exception as e:
        print(f"Error loading ASR model: {e}")
        transcriber = None

# --- 2. Transcription and Plotting Function ---
def transcribe_and_plot_audio_chunk(audio_data, gain_reduction_factor=1.0):
    """
    Transcribes an audio chunk, applies gain reduction, and generates a waveform plot.

    Args:
        audio_data: A tuple (sample_rate, numpy_array).
        gain_reduction_factor (float): Factor to reduce audio gain. 1.0 means no change.

    Returns:
        A tuple (transcribed_text, matplotlib_figure_object).
    """
    global transcriber
    if transcriber is None:
        return "ASR model not loaded.", None
    if audio_data is None:
        return "", None

    try:
        sample_rate, data_raw = audio_data
        
        # --- Essential Audio Processing ---
        # 1. Convert to float32 and normalize if it's int16 (or other int types)
        if np.issubdtype(data_raw.dtype, np.integer):
            data_float = data_raw.astype(np.float32) / np.iinfo(data_raw.dtype).max
        elif data_raw.dtype == np.float32:
            data_float = data_raw.copy()
        else:
            data_float = data_raw.astype(np.float32) # Fallback

        # 2. Apply Software Gain Reduction
        print(f"  Applying gain reduction factor: {gain_reduction_factor}")
        data_float_reduced = data_float * gain_reduction_factor

        # 3. Convert to mono if stereo (Whisper generally expects mono)
        if len(data_float_reduced.shape) > 1 and data_float_reduced.shape[1] > 1:
            data_mono = np.mean(data_float_reduced, axis=1)
        else:
            data_mono = data_float_reduced
        
        # 4. Clipping (ensure data is within [-1.0, 1.0])
        data_processed = np.clip(data_mono, -1.0, 1.0)
        
        # --- ASR Transcription ---
        transcription_result = transcriber({"sampling_rate": sample_rate, "raw": data_processed})
        text = transcription_result.get("text", "No text in result").strip()
        print(f"  ASR Transcription: {text}")

        # --- Waveform Plotting ---
        fig, ax = plt.subplots(figsize=(6, 2.5)) # Adjust figsize as needed
        time_axis = np.linspace(0, len(data_processed) / sample_rate, num=len(data_processed))
        ax.plot(time_axis, data_processed, lw=0.5) # lw is line width
        ax.set_title("Audio Waveform (Last Chunk)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim([-1.05, 1.05]) # Fixed Y-axis for consistent scaling
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        
        return text, fig

    except Exception as e:
        print(f"Error during transcription/plotting: {e}")
        import traceback
        traceback.print_exc()
        # Create an empty plot on error to avoid Gradio issues
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Error generating plot", ha='center', va='center')
        plt.tight_layout()
        return f"Error: {str(e)}", fig


# --- 3. Gradio Interface ---
def continuous_transcription_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # 🎤 Continuous ASR Recording & Waveform Display
            Speak into your microphone. Transcription will appear below.
            The waveform of the last processed audio chunk is displayed.
            Use the slider to apply software gain reduction if the input is too loud.
            **Note:** For persistent high gain, check your **Operating System microphone settings** first!
            """
        )

        full_transcript_var = gr.State("")

        with gr.Row():
            with gr.Column(scale=2):
                output_textbox = gr.Textbox(
                    label="Live Transcription",
                    lines=10,
                    interactive=False,
                    autoscroll=True
                )
            with gr.Column(scale=1):
                plot_output = gr.Plot(label="Audio Waveform (Last Chunk)")
                gain_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                    label="Software Gain Reduction Factor",
                    info="Adjust to reduce volume if audio is too loud (1.0 = no reduction)"
                )

        mic_input_stream = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=True,
            label="Microphone Input (Streaming)",
        )

        def update_transcript_and_plot_stream(new_audio_chunk, current_transcript, gain_factor):
            if new_audio_chunk is None:
                return current_transcript, current_transcript, None # No change to plot if no new audio

            print("Received audio chunk for streaming...")
            chunk_text, fig = transcribe_and_plot_audio_chunk(new_audio_chunk, gain_factor)

            if chunk_text and not chunk_text.startswith("Error:"):
                updated_transcript = current_transcript + (" " if current_transcript else "") + chunk_text
                return updated_transcript, updated_transcript, fig
            elif chunk_text.startswith("Error:"):
                print(f"ASR Error on chunk: {chunk_text}")
                return current_transcript, current_transcript, fig # Show error plot
            return current_transcript, current_transcript, fig # Show plot even if no new text


        mic_input_stream.stream(
            fn=update_transcript_and_plot_stream,
            inputs=[mic_input_stream, full_transcript_var, gain_slider],
            outputs=[full_transcript_var, output_textbox, plot_output],
            show_progress="hidden"
        )

        def clear_all():
            # Create a blank plot to clear the existing one
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "Cleared", ha='center', va='center')
            plt.tight_layout()
            return "", "", fig

        clear_button = gr.Button("Clear Transcript & Plot")
        clear_button.click(
            fn=clear_all,
            inputs=None, # No direct inputs needed from UI for this function
            outputs=[full_transcript_var, output_textbox, plot_output]
        )
        
        gr.Markdown("---")
        gr.Markdown("### How to Interpret the Waveform:")
        gr.Markdown(
            "- **Ideal:** The waveform should fluctuate, showing peaks and valleys (dynamic range). Peaks should generally stay below +/- 0.8 or +/- 0.9 to avoid clipping and allow headroom.\n"
            "- **Too Loud / Clipped:** If the waveform looks like a solid block pressed against the top and bottom (+1.0 and -1.0), it's clipped. The audio information is lost.\n"
            "- **Too Quiet:** If the waveform is very flat and close to the zero line, it's too quiet (unlikely your issue now).\n"
            "Use the **Gain Reduction Factor** slider to reduce the volume *before* it goes to the ASR. If the waveform is clipped *before* software reduction, the damage is already done, so adjusting OS mic settings is key."
        )

    return demo

# --- Run the App ---
if __name__ == "__main__":
    load_asr_model() # Load the model once at startup
    if transcriber is None:
        print("Halting app launch as ASR model failed to load.")
    else:
        app = continuous_transcription_app()
        print("Launching Gradio app...")
        # Use share=True if you want to create a temporary public link
        app.launch(debug=True, share=False)