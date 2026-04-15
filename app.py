import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
WEIGHTS_DIR = BASE_DIR / "weights"
DEFAULT_MODEL_PATH = WEIGHTS_DIR / "best.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Файл модели не найден: {}\n"
            "Положи best.pt в папку weights или укажи MODEL_PATH.".format(MODEL_PATH)
        )
    return YOLO(str(MODEL_PATH))


app = FastAPI(title="NEU Defect Detection")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model = load_model()


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def draw_detections(
    image_bgr,
    boxes,
    class_ids,
    confs,
    names_map
):
    image = image_bgr.copy()

    for box, cls_id, conf in zip(boxes, class_ids, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 0)
        label = "{}: {:.2f}".format(names_map.get(int(cls_id), str(cls_id)), float(conf))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return image


def run_inference(image_path: Path) -> Dict[str, Any]:
    results = model.predict(
        source=str(image_path),
        conf=0.25,
        iou=0.45,
        save=False,
        verbose=False
    )

    result = results[0]
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError("Не удалось прочитать изображение: {}".format(image_path))

    detections = []
    names_map = result.names if hasattr(result, "names") else {i: n for i, n in enumerate(CLASS_NAMES)}

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        annotated = draw_detections(image_bgr, boxes, class_ids, confs, names_map)

        for box, cls_id, conf in zip(boxes, class_ids, confs):
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append({
                "class_id": int(cls_id),
                "class_name": names_map.get(int(cls_id), str(cls_id)),
                "confidence": round(float(conf), 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })
    else:
        annotated = image_bgr.copy()

    result_name = "result_{}.jpg".format(uuid.uuid4().hex)
    result_path = RESULT_DIR / result_name
    cv2.imwrite(str(result_path), annotated)

    status = "Дефекты обнаружены" if len(detections) > 0 else "Дефекты не обнаружены"

    return {
        "result_image_url": "/results/{}".format(result_name),
        "detections": detections,
        "status": status,
        "detections_count": len(detections),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_image_url": None,
            "detections": [],
            "status": None,
            "error": None,
            "detections_count": 0,
            "model_path": str(MODEL_PATH),
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    error = None
    result_image_url = None
    detections = []
    status = None
    detections_count = 0

    try:
        if file is None or not file.filename:
            raise ValueError("Файл не был выбран.")

        if not allowed_file(file.filename):
            raise ValueError("Поддерживаются только изображения: jpg, jpeg, png, bmp, webp, tif, tiff.")

        ext = Path(file.filename).suffix.lower()
        upload_name = "upload_{}{}".format(uuid.uuid4().hex, ext)
        upload_path = UPLOAD_DIR / upload_name

        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)

        inference = run_inference(upload_path)
        result_image_url = inference["result_image_url"]
        detections = inference["detections"]
        status = inference["status"]
        detections_count = inference["detections_count"]

    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_image_url": result_image_url,
            "detections": detections,
            "status": status,
            "error": error,
            "detections_count": detections_count,
            "model_path": str(MODEL_PATH),
        }
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "app": "NEU Defect Detection",
        "model_path": str(MODEL_PATH),
        "class_names": CLASS_NAMES,
        "python_38_compatible": True,
    }