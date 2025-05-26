import gradio as gr
from transformers import pipeline
import numpy as np # Still useful for audio data handling if needed
import torch

# --- Configuration ---
ASR_MODEL_NAME = "openai/whisper-base.en" # Using a more descriptive name
ASR_PIPELINE = None

# --- 1. Load ASR Model ---
def load_asr_pipeline():
    global ASR_PIPELINE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        print(f"Loading ASR pipeline for model: {ASR_MODEL_NAME}...")
        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=device
        )
        print(f"ASR pipeline loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading ASR pipeline: {e}")
        ASR_PIPELINE = None

# --- 2. Transcription Function ---
def transcribe_audio_file(audio_input):
    """
    Transcribes an audio segment.
    Args:
        audio_input: Can be a filepath string (from gr.Audio type="filepath")
                     or a tuple (sample_rate, numpy_array) (from gr.Audio type="numpy").
    Returns:
        The transcribed text (str).
    """
    if ASR_PIPELINE is None:
        return "ASR model not loaded. Please check server logs."
    
    if audio_input is None:
        return "No audio input received. Please record or upload audio."

    print(f"Received audio_input for transcription. Type: {type(audio_input)}")

    try:
        # The Hugging Face pipeline can often handle filepaths directly.
        # If audio_input is (sample_rate, data_array) from type="numpy",
        # it can also often handle {"sampling_rate": sr, "raw": data}.
        # For simplicity with type="filepath", audio_input will be the path.
        
        # If you chose type="numpy" for gr.Audio, you would handle it like this:
        # if isinstance(audio_input, tuple):
        #     sample_rate, data_array = audio_input
        #     print(f"Processing numpy audio. SR: {sample_rate}, Shape: {data_array.shape}, Dtype: {data_array.dtype}")
        #     # Ensure data is float32 if it's int16
        #     if np.issubdtype(data_array.dtype, np.integer):
        #         data_array = data_array.astype(np.float32) / np.iinfo(data_array.dtype).max
        #     elif data_array.dtype != np.float32:
        #         data_array = data_array.astype(np.float32)
        #     # Pass raw data to the pipeline
        #     result = ASR_PIPELINE({"sampling_rate": sample_rate, "raw": data_array})
        # else: # Assuming filepath
        #     result = ASR_PIPELINE(audio_input)

        # For type="filepath" (which is often simpler as user mentioned "access the file")
        print(f"Transcribing audio file: {audio_input}")
        result = ASR_PIPELINE(audio_input)
        
        transcribed_text = result.get("text", "Transcription error or empty audio.")
        print(f"Transcription: {transcribed_text}")
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during transcription: {str(e)}"

# --- 3. Gradio Interface ---
def simple_asr_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # Simple Speech-to-Text App
            1. Click "Record from microphone".
            2. Speak your message.
            3. Click "Stop recording".
            4. The transcription will appear below.
            (Uses Whisper model: `{ASR_MODEL_NAME}`)
            """
        )

        # Audio input component
        # type="filepath" means the function will receive a path to a temporary audio file.
        # type="numpy" means the function will receive (sample_rate, np.ndarray).
        # For this simple version, "filepath" is straightforward.
        audio_input_mic = gr.Audio(
            sources=["microphone"], # Prioritize microphone
            # sources=["microphone", "upload"], # To allow both recording and uploading
            type="filepath",        # Output type received by the Python function
            label="Record or Upload Audio",
            # waveform_options=gr.WaveformOptions(waveform_color="#0176dd", waveform_progress_color="#0056a3") # Optional styling
        )

        # Textbox for displaying the transcription
        transcription_output_textbox = gr.Textbox(
            label="Transcription",
            lines=5,
            interactive=False # User should not edit this directly
        )

        # Button to trigger transcription (alternative to using event listeners on Audio)
        # transcribe_button = gr.Button("Transcribe Audio")

        # Link the audio component's "stop_recording" or "upload" event to the transcription function
        # The 'change' event fires when a new audio file is present (either recorded or uploaded).
        # The 'stop_recording' event is specific to when microphone recording is stopped.
        # The 'upload' event is specific to when a file is uploaded.
        # Using 'change' is often a good catch-all if you allow both mic and upload.
        # If only microphone, 'stop_recording' is very clear.

        # Let's use 'stop_recording' for microphone and 'upload' if you also allow file uploads.
        # For simplicity, if only microphone is the primary source, 'stop_recording' is fine.
        # If you also have 'upload' in sources, you might need two listeners or use 'change'.

        # The `gr.Audio` component will trigger its processing function (linked via an event)
        # when the recording is stopped or a file is uploaded.
        # We can use the .change() event for type="filepath" as it covers both.
        audio_input_mic.change(
            fn=transcribe_audio_file,
            inputs=[audio_input_mic],
            outputs=[transcription_output_textbox],
            show_progress="minimal"
        )
        
        # If you wanted a separate button:
        # transcribe_button.click(
        #     fn=transcribe_audio_file,
        #     inputs=[audio_input_mic],
        #     outputs=[transcription_output_textbox],
        #     show_progress="minimal"
        # )
        
        gr.Markdown("---")
        gr.Markdown("Powered by [Gradio](httpsai.com/gradio-docs) and [Hugging Face Transformers](https://huggingface.co/docs/transformers/index).")

    return demo

# --- Run the App ---
if __name__ == "__main__":
    load_asr_pipeline() # Load the ASR model once at startup
    if ASR_PIPELINE is None:
        print("Halting app launch as ASR pipeline failed to load.")
    else:
        app = simple_asr_app()
        print("Launching Simple ASR Gradio app...")
        # Set share=True if you want to create a temporary public link to share with others
        app.launch(debug=True, share=False)