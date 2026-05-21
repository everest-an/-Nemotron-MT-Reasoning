import os
import shutil
import zipfile

def package_submission(lora_dir="./submission/adapter", output_zip="submission.zip"):
    """
    Kaggle Nvidia Nemotron Reasoning Challenge
    要求提交包含 LoRA 权重的 ZIP 文件。包含 adapter_config.json 以及 safetensors (或 bin) 文件。
    确保 Rank <= 32。
    """
    if not os.path.exists(lora_dir):
        print(f"Error: LoRA directory {lora_dir} does not exist. Please run training first.")
        return

    print(f"Packaging {lora_dir} into {output_zip}...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(lora_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, lora_dir)
                zipf.write(file_path, arcname)
                print(f"  Added {arcname}")
                
    print(f"Successfully packaged {output_zip} for Kaggle submission!")

if __name__ == "__main__":
    package_submission()
