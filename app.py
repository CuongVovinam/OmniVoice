#!/usr/bin/env python3
import os
import torch
import gradio as gr
import numpy as np
import zipfile
import tempfile
import shutil
import logging
from typing import List, Optional, Tuple, Dict, Any
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import lang_display_name

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global model variable
model = None

def load_model(checkpoint="k2-fsa/OmniVoice"):
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model from {checkpoint} on {device}...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        load_asr=True
    )
    logging.info("Model loaded successfully.")
    return model

# ---------------------------------------------------------------------------
# Voice Design categories (Simplified)
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "English Accent": [
        "American Accent", "Australian Accent", "British Accent", "Chinese Accent",
        "Canadian Accent", "Indian Accent", "Korean Accent", "Portuguese Accent",
        "Russian Accent", "Japanese Accent"
    ],
}

LANG_MAP = {
    "English": "en",
    "Vietnamese": "vi"
}

def generate_audio(
    text: str,
    language: str,
    num_step: int,
    guidance_scale: float,
    denoise: bool,
    speed: float,
    duration: Optional[float],
    preprocess_prompt: bool,
    postprocess_output: bool,
    mode: str,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
) -> Tuple[int, np.ndarray]:
    
    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step),
        guidance_scale=float(guidance_scale),
        denoise=bool(denoise),
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang_code = LANG_MAP.get(language)
    
    kw = {
        "text": text.strip(),
        "language": lang_code,
        "generation_config": gen_config
    }

    if speed != 1.0:
        kw["speed"] = float(speed)
    if duration and duration > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            raise ValueError("Reference audio is required for Voice Clone.")
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text if ref_text else None,
        )

    if instruct:
        kw["instruct"] = instruct

    audio = model.generate(**kw)
    sampling_rate = model.sampling_rate
    waveform = (audio[0] * 32767).astype(np.int16)
    return sampling_rate, waveform

import scipy.io.wavfile as wavfile

def process(
    input_mode: str,
    single_text: str,
    batch_files: Optional[List[Any]],
    language: str,
    num_step: int,
    guidance_scale: float,
    denoise: bool,
    speed: float,
    duration: Optional[float],
    preprocess_prompt: bool,
    postprocess_output: bool,
    mode: str,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    *design_args
):
    if model is None:
        return None, None, "Model not loaded."

    instruct = None
    if mode == "design":
        # Build instruct from design_args
        selected = [arg for arg in design_args if arg and arg != "Auto"]
        if selected:
            instruct = ", ".join(selected)

    try:
        if input_mode == "Single Text":
            if not single_text.strip():
                return None, None, "Please enter text."
            
            sr, waveform = generate_audio(
                single_text, language, num_step, guidance_scale, denoise, speed, 
                duration, preprocess_prompt, postprocess_output, mode, ref_audio, ref_text, instruct
            )
            
            # Save to temporary file for preview and download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wavfile.write(temp_file.name, sr, waveform)
            return (sr, waveform), temp_file.name, "Done."

        else: # Batch Files
            if not batch_files:
                return None, None, "Please upload .txt files."
            
            temp_dir = tempfile.mkdtemp()
            output_files = []
            
            for file_obj in batch_files:
                file_path = file_obj.name
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                
                sr, waveform = generate_audio(
                    content, language, num_step, guidance_scale, denoise, speed, 
                    duration, preprocess_prompt, postprocess_output, mode, ref_audio, ref_text, instruct
                )
                
                out_path = os.path.join(temp_dir, f"{base_name}.wav")
                wavfile.write(out_path, sr, waveform)
                output_files.append(out_path)
            
            if not output_files:
                return None, None, "No valid content found in files."
            
            # Create ZIP
            zip_name = "generated_audios.zip"
            zip_path = os.path.join(tempfile.gettempdir(), zip_name)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for f in output_files:
                    zipf.write(f, os.path.basename(f))
            
            shutil.rmtree(temp_dir)
            return None, zip_path, f"Processed {len(output_files)} files. Download ZIP below."

    except Exception as e:
        logging.error(f"Error during generation: {e}")
        return None, None, f"Error: {str(e)}"

# =====================================================================
# UI
# =====================================================================

def build_app():
    with gr.Blocks(title="OmniVoice Optimized") as app:
        gr.Markdown("# OmniVoice Optimized (EN/VI)")
        gr.Markdown("High-speed TTS with Batch Processing support.")

        with gr.Tabs():
            # VOICE CLONE TAB
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column():
                        vc_input_mode = gr.Radio(["Single Text", "Batch Text Files"], label="Input Mode", value="Single Text")
                        vc_text = gr.Textbox(label="Text to Synthesize", lines=3, visible=True)
                        vc_files = gr.File(label="Upload .txt Files", file_count="multiple", file_types=[".txt"], visible=False)
                        
                        vc_ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                        vc_ref_text = gr.Textbox(label="Reference Text (Optional)", placeholder="Transcribe automatically if empty")
                        
                        vc_lang = gr.Dropdown(["English", "Vietnamese"], label="Language", value="English")
                        
                        with gr.Accordion("Generation Settings", open=False):
                            vc_num_step = gr.Slider(4, 64, value=24, step=1, label="Inference Steps (Lower = Faster)")
                            vc_guidance = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale")
                            vc_denoise = gr.Checkbox(label="Denoise", value=True)
                            vc_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                            vc_duration = gr.Number(label="Fixed Duration (seconds)", value=None)
                            vc_pp = gr.Checkbox(label="Preprocess Prompt", value=True)
                            vc_po = gr.Checkbox(label="Postprocess Output", value=True)
                        
                        vc_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column():
                        vc_audio_preview = gr.Audio(label="Audio Preview", type="numpy")
                        vc_file_output = gr.File(label="Download Output (WAV or ZIP)")
                        vc_status = gr.Textbox(label="Status", interactive=False)

                # Toggle visibility
                vc_input_mode.change(
                    fn=lambda mode: [gr.update(visible=mode == "Single Text"), gr.update(visible=mode == "Batch Text Files")],
                    inputs=vc_input_mode,
                    outputs=[vc_text, vc_files]
                )

            # VOICE DESIGN TAB
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column():
                        vd_input_mode = gr.Radio(["Single Text", "Batch Text Files"], label="Input Mode", value="Single Text")
                        vd_text = gr.Textbox(label="Text to Synthesize", lines=3, visible=True)
                        vd_files = gr.File(label="Upload .txt Files", file_count="multiple", file_types=[".txt"], visible=False)
                        
                        vd_lang = gr.Dropdown(["English", "Vietnamese"], label="Language", value="English")
                        
                        vd_design_dropdowns = []
                        for cat, choices in _CATEGORIES.items():
                            vd_design_dropdowns.append(gr.Dropdown(["Auto"] + choices, label=cat, value="Auto"))
                        
                        with gr.Accordion("Generation Settings", open=False):
                            vd_num_step = gr.Slider(4, 64, value=24, step=1, label="Inference Steps")
                            vd_guidance = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale")
                            vd_denoise = gr.Checkbox(label="Denoise", value=True)
                            vd_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                            vd_duration = gr.Number(label="Fixed Duration (seconds)", value=None)
                            vd_pp = gr.Checkbox(label="Preprocess Prompt", value=True)
                            vd_po = gr.Checkbox(label="Postprocess Output", value=True)
                        
                        vd_btn = gr.Button("Generate", variant="primary")

                    with gr.Column():
                        vd_audio_preview = gr.Audio(label="Audio Preview", type="numpy")
                        vd_file_output = gr.File(label="Download Output (WAV or ZIP)")
                        vd_status = gr.Textbox(label="Status", interactive=False)

                # Toggle visibility
                vd_input_mode.change(
                    fn=lambda mode: [gr.update(visible=mode == "Single Text"), gr.update(visible=mode == "Batch Text Files")],
                    inputs=vd_input_mode,
                    outputs=[vd_text, vd_files]
                )

        # Event Handlers
        vc_btn.click(
            fn=process,
            inputs=[
                vc_input_mode, vc_text, vc_files, vc_lang, vc_num_step, vc_guidance, 
                vc_denoise, vc_speed, vc_duration, vc_pp, vc_po, gr.State("clone"), 
                vc_ref_audio, vc_ref_text
            ],
            outputs=[vc_audio_preview, vc_file_output, vc_status]
        )

        vd_btn.click(
            fn=process,
            inputs=[
                vd_input_mode, vd_text, vd_files, vd_lang, vd_num_step, vd_guidance, 
                vd_denoise, vd_speed, vd_duration, vd_pp, vd_po, gr.State("design"),
                gr.State(None), gr.State(None) # ref_audio, ref_text
            ] + vd_design_dropdowns,
            outputs=[vd_audio_preview, vd_file_output, vd_status]
        )

    return app

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the optimized OmniVoice app.")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the app on (default: 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to run the app on (default: 0.0.0.0)")
    parser.add_argument("--model", type=str, default="k2-fsa/OmniVoice", help="Path to the model checkpoint or HF repo ID")
    args = parser.parse_args()

    load_model(args.model)
    app = build_app()
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    app.launch(server_name=args.host, server_port=args.port, theme=theme)
