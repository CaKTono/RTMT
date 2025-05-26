# minimal_test.py
import gradio as gr
# import numpy as np # Not strictly needed for this minimal test if dummy_stream_fn doesn't use it

def dummy_stream_fn(audio_chunk_from_mic):
    print("Minimal dummy stream function received a chunk.")
    # Example of how you might inspect, but keep it simple for the test:
    # if audio_chunk_from_mic is not None:
    #     try:
    #         sample_rate, data = audio_chunk_from_mic
    #         print(f"Chunk details: SR={sample_rate}, Data shape={data.shape if data is not None else 'None'}")
    #     except Exception as e:
    #         print(f"Error unpacking audio chunk: {e}")
    # else:
    #     print("Audio chunk was None")
    return "dummy output"

with gr.Blocks() as demo_minimal:
    gr.Markdown("Absolute Minimal Audio Stream Test")
    mic_input = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Mic")
    text_output = gr.Textbox(label="Output")

    mic_input.stream(fn=dummy_stream_fn, inputs=[mic_input], outputs=[text_output])

print("Attempting to launch minimal Gradio app...")
try:
    demo_minimal.launch(
        server_name="0.0.0.0",
        server_port=7861, # Using a different port
        debug=True
    )
except Exception as e:
    print(f"An error occurred during minimal Gradio launch: {e}")
    if isinstance(e, TypeError) and "argument of type 'bool' is not iterable" in str(e):
        print("\nThis TypeError is present even in the absolute minimal example.")