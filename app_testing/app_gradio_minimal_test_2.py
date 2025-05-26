# app_gradio_minimal_stream_numpy_test.py
import gradio as gr
import numpy as np
import time

print("Minimal App Test: Starting Streaming with NumPy...")
print(f"Gradio version being used: {gr.__version__}")
print(f"NumPy version being used: {np.__version__}")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ultra-Minimal Gradio Test - Stage: Minimal Stream with NumPy")
    gr.Markdown("Objective: Test gr.Audio(type='numpy', streaming=True).stream()")

    # --- UI Components ---
    print("Defining UI components for NumPy streaming test...")
    audio_input_mic = gr.Audio(
        sources=["microphone"],
        type="numpy",       # CHANGED to numpy
        streaming=True,     # CHANGED to True
        label="Live Microphone Input (Streaming NumPy)",
        # You can experiment with stream_every if needed, e.g., stream_every=1 (for 1-second chunks)
        # default is 0.1 seconds (100ms)
    )
    
    stream_output_tb = gr.Textbox(
        label="Stream Output (NumPy Chunk Info)", lines=5, interactive=False, autoscroll=True
    )
    print("  SUCCESS: Minimal UI components for NumPy stream test defined.")

    # --- Event Handlers ---
    def numpy_stream_fn(audio_chunk): # Expects (sample_rate, numpy_array) or None
        if audio_chunk is not None:
            sample_rate, data = audio_chunk
            # Let's print info about the received chunk
            # In a real app, you'd process this 'data'
            info = (
                f"Received audio chunk:\n"
                f"  Sample Rate: {sample_rate}\n"
                f"  Data Type: {data.dtype}\n"
                f"  Data Shape: {data.shape}\n"
                f"  Data Min: {np.min(data) if data.size > 0 else 'N/A'}\n"
                f"  Data Max: {np.max(data) if data.size > 0 else 'N/A'}\n"
                f"  Data Mean: {np.mean(data) if data.size > 0 else 'N/A'}"
            )
            print(info) # Print to console for more detail
            return info # Display basic info in the textbox
        return "No audio chunk received or stream initializing..."

    print("Defining audio_input_mic.stream event handler for NumPy...")
    audio_input_mic.stream(
        fn=numpy_stream_fn,
        inputs=[audio_input_mic],       # The streaming audio component itself
        outputs=[stream_output_tb],     # Output to the textbox
        # show_progress="hidden" # Optional: uncomment to hide "Working..." indicator on UI for smoother look
    )
    print("SUCCESS: audio_input_mic.stream event handler for NumPy defined.")

    # Add a button to clear the textbox if you want
    def clear_text():
        return ""
    clear_btn = gr.Button("Clear Output")
    clear_btn.click(fn=clear_text, inputs=None, outputs=[stream_output_tb])


if __name__ == "__main__":
    print("Attempting to launch Gradio demo with NumPy streaming...")
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
    except Exception as e:
        print(f"Error launching Gradio demo with NumPy stream: {e}")
        # Print version info again in case of error
        import sys
        print(f"Python version: {sys.version}")
        try:
            import pydantic; print(f"Pydantic version: {pydantic.__version__}")
            import fastapi; print(f"FastAPI version: {fastapi.__version__}")
            # ... (other dependencies if relevant)
        except ImportError as ie:
            print(f"Could not import a core dependency: {ie}")