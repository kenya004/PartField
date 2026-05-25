from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
import uvicorn
import subprocess
import glob

app = FastAPI()


BASE_DIR = "/home/mrkab/git/PartField" 
SAVE_DIR = os.path.join(BASE_DIR, "data/objaverse_samples")
PYTHON_EXE = "/home/mrkab/miniconda3/envs/partfield/bin/python"

RESULT_ZIP_DIR = os.path.join(BASE_DIR, "result_zip")
os.makedirs(RESULT_ZIP_DIR, exist_ok=True)

@app.post("/upload_glb")
async def upload_glb(file: UploadFile = File(...)):
    try:

        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        file_location = os.path.join(SAVE_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        subprocess.run([
            PYTHON_EXE, "partfield_inference.py",
            "-c", "configs/final/demo.yaml",
            "--opts", "continue_ckpt", "model/model_objaverse.ckpt",
            "result_name", "partfield_features/objaverse",
            "dataset.data_path", "data/objaverse_samples"
        ], cwd=BASE_DIR, check=True)

        subprocess.run([
            PYTHON_EXE, "run_part_clustering.py",
            "--root", "exp_results/partfield_features/objaverse",
            "--dump_dir", "exp_results/clustering/objaverse",
            "--source_dir", "data/objaverse_samples",
            "--use_agglo", "True",
            "--max_num_clusters", "2",
            "--option", "0"
        ], cwd=BASE_DIR, check=True)

        cluster_dir_pattern = os.path.join(BASE_DIR, "exp_results/clustering/objaverse/ply/*_clusters")
        cluster_dirs = glob.glob(cluster_dir_pattern)

        if cluster_dirs:
            latest_dir = max(cluster_dirs, key=os.path.getmtime)
            
       
            zip_base_path = os.path.join(RESULT_ZIP_DIR, "parts_split")
            
            if os.path.exists(f"{zip_base_path}.zip"):
                os.remove(f"{zip_base_path}.zip")
                
            shutil.make_archive(zip_base_path, 'zip', latest_dir)
            
            return FileResponse(
                path=f"{zip_base_path}.zip",
                filename="parts_split.zip",
                media_type='application/zip'
            )
        else:
            return {"status": "error", "message": "Cluster directory not found."}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Command failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8003)