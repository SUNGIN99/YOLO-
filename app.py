from fastapi import FastAPI, File, UploadFile
import os
import json
import subprocess
from fastapi.responses import JSONResponse

app = FastAPI()

WEIGHTS_PATH = './car_license/runs/train/license_plate_fine_tuned/weights/best.pt'
DETECT_SCRIPT = 'python ./car_license/detect.py'

@app.post("/detect")
async def detect_image(image: UploadFile = File(...)):
    try:        
        image_path = os.path.join('uploads', image.filename)
        if os.path.exists(image_path):
            os.remove(image_path)

        with open(image_path, 'wb') as f:
            f.write(await image.read())

        # python ./car_license/detect.py --source ./car_license/testdata/truck3.jpg --weights ./car_license/runs/train/license_plate_fine_tuned/weights/best.pt --save-crop
        command = f"{DETECT_SCRIPT} --weights {WEIGHTS_PATH} --source {image_path} --save-crop"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)

        #print("Process : ", process)
        #print("Process code : ", process.returncode)
        data = json.loads(process.stdout)
        print("[From Yolo] Detected Path : ", data["results"]["crop_dir"])

        platePath = data["results"]["crop_dir"]

        ## OCR 수행 서드파티 로직 추가
        command = ["python", "car_ocr.py", platePath]
        car_ocr = subprocess.run(command, capture_output=True, text=True)

        data = json.loads(car_ocr.stdout)
        carPlate = data["results"]
        print("[From Ocr] Detected Plate : ", str(carPlate))

        if process.returncode == 0 and car_ocr.returncode == 0:
            return {
                "message": "Detection completed successfully.",
                "data" : carPlate
                }
        else:
            return JSONResponse(content={"error": "Detection failed.", "details": process.stderr | car_ocr.stderr }, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/test")
def getTest():
    try:
        return {"test" : "complet"}
    except Exception as e:
        return {"Err" : e}