# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
sys.path.extend(['/vseq2vseq'])
import subprocess

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Folder locations that are relative to the root of the cog repo
        os.makedirs("/root/.cache/huggingface/hub/")
        sys_cache = "/root/.cache/huggingface/hub/"
        local_sdxl = "/src/model-cache/"
        sdxl_model = "models--stabilityai--stable-diffusion-xl-base-1.0"
        vseq2vseq_model = "models--motexture--vseq2vseq"
        # Create a symlink to the 2 models
        os.system("ln -s "+local_sdxl+sdxl_model+" "+sys_cache+sdxl_model)
        os.system("ln -s "+local_sdxl+vseq2vseq_model+" "+sys_cache+vseq2vseq_model)
        
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A stormtrooper surfing on the ocean"
        ),
        guidance_scale: int = Input(
            description="Guidance scale",
            default=20
        ),
        image_guidance_scale: int = Input(
            description="Individually scale the image guidance",
            default=12
        ),
        fps: int = Input(
            description="Frames per second",
            default=16
        ),
        number_of_frames: int = Input(
            description="Number of frames",
            default=24
        ),
        width: int = Input(
            description="Width",
            default=384
        ),
        height: int = Input(
            description="Height",
            default=192
        ),
        img_width: int = Input(
            description="Image width",
            default=1152
        ),
        img_height: int = Input(
            description="Image height",
            default=640
        ),
        num_steps: int = Input(
            description="Number of steps",
            default=30
        ),
        times: int = Input(
            description="Times",
            default=8
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Cleanup past runs
        os.system("rm -rf /vseq2vseq/output")
        os.makedirs("/vseq2vseq/output")
        
        command = [
            "python", "/vseq2vseq/inference.py",
            "--prompt", prompt,
            "--model", "motexture/vseq2vseq",
            "--model-2d", "stabilityai/stable-diffusion-xl-base-1.0",
            "--guidance-scale", str(guidance_scale),
            "--image-guidance-scale", str(image_guidance_scale),
            "--fps", str(fps),
            "--sdp",
            "--num-frames", str(number_of_frames),
            "--width", str(width),
            "--height", str(height),
            "--image-width", str(img_width),
            "--image-height", str(img_height),
            "--num-steps", str(num_steps),
            "--times", str(times),
            "--min-conditioning-n-sample-frames", "2",
            "--max-conditioning-n-sample-frames", "2",
            "--device", "cuda",
            "--save-init",
            "--include-model"
        ]
        subprocess.run(command)

        output_dir = "/src/output/"
        output_path = None
        # Find an mp4 file in the "output" directory
        for file in os.listdir(output_dir):
            if file.endswith(".mp4"):
                output_path = str(output_dir + file)
                print(output_path)
                break

        return Path(output_path)
