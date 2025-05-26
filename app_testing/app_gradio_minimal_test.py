# app_gradio_minimal_test.py
import gradio as gr
import numpy as np
import time

print("Minimal App Test: Starting...")
print(f"Gradio version being used: {gr.__version__}")

class MinimalConfig: # Not really used in this version
    SUPPORTED_LANGUAGES = {"en": "English", "zh": "Chinese"}
config = MinimalConfig()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ultra-Minimal Gradio Test - Stage: Minimal Stream")
    gr.Markdown("Objective: Test gr.Audio().stream() with minimal inputs/outputs.")

    # --- State Variables (defined but NOT used in the stream inputs/outputs for this test) ---
    print("Defining state variables (but not all used in this specific stream test)...")
    transcript_history_state = gr.State([])
    translation_history_state = gr.State([])
    last_vad_time_gr_state = gr.State(time.time())
    vad_speech_active_gr_state = gr.State(0)
    audio_buffer_state = gr.State(None)
    print("  SUCCESS: All state variables defined.")

    # --- UI Components ---
    print("Defining UI components...")
    audio_input_mic = gr.Audio(
        sources=["microphone"],
        type="filepath", # CHANGE THIS
        streaming=False,
        label="Live Microphone Input",
    )
    # Only one output textbox for this minimal stream test
    transcript_output_tb = gr.Textbox(
        label="Stream Output", lines=3, interactive=False, autoscroll=True
    )
    print("  SUCCESS: Minimal UI components for stream test defined.")

    # --- Event Handlers ---
    def ultra_dummy_stream_fn(audio_filepath): # Expects a filepath string
        # print(f"Ultra dummy stream called. Audio filepath: {audio_filepath}, Type: {type(audio_filepath)}")
        if audio_filepath is not None:
            # In a real app, you would load the audio from this temp filepath:
            # e.g., import soundfile as sf
            # data, samplerate = sf.read(audio_filepath)
            return f"Audio filepath received: {audio_filepath}"
        return "No audio filepath received in this chunk."

    print("Defining ultra-minimal audio_input_mic.stream event handler...")
    # audio_input_mic.stream(
    #     fn=ultra_dummy_stream_fn,
    #     inputs=[audio_input_mic],       # ONLY the audio component itself as input
    #     outputs=[transcript_output_tb]  # ONLY a single Textbox as output
    # )
    print("SUCCESS: Ultra-minimal audio_input_mic.stream event handler defined.")

if __name__ == "__main__":
    print("Attempting to launch Gradio demo with ultra-minimal stream handler...")
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
    except Exception as e:
        print(f"Error launching Gradio demo with ultra-minimal stream: {e}")
        if "TypeError: argument of type 'bool' is not iterable" in str(e):
            print(">>> The TypeError PERSISTS even with an ultra-minimal stream setup! <<<")
            print("This strongly points to an issue with gr.Audio(type='numpy', streaming=True).stream()")
            print("in combination with your Gradio version / environment (possibly NumPy 2.0.2).")
        import sys
        print(f"Python version: {sys.version}")
        try:
            import pydantic; print(f"Pydantic version: {pydantic.__version__}")
            import fastapi; print(f"FastAPI version: {fastapi.__version__}")
            import starlette; print(f"Starlette version: {starlette.__version__}")
            import uvicorn; print(f"Uvicorn version: {uvicorn.__version__}")
            print(f"Numpy version: {np.__version__}") # Make sure np is imported
        except ImportError as ie:
            print(f"Could not import a core dependency: {ie}")