from flask import Flask, request, render_template, jsonify
from deepface import DeepFace
import mysql.connector
import numpy as np
import pickle
import base64
import os
import cv2
from datetime import datetime
from config import MODEL_CACHE_DIR
import tensorflow as tf
from dotenv import load_dotenv

# Load environment variables dari .env file
load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Set home dir untuk model cache
os.environ['DEEPFACE_HOME'] = MODEL_CACHE_DIR

# Global model instances
tflite_fp16_interpreter = None
tflite_fp16_available = False

def load_tflite_fp16_model():
    """Load TFLite FP16 quantized model"""
    global tflite_fp16_interpreter, tflite_fp16_available
    try:
        tflite_fp16_path = "models/arcface_fp16.tflite"
        if os.path.exists(tflite_fp16_path):
            tflite_fp16_interpreter = tf.lite.Interpreter(model_path=tflite_fp16_path)
            tflite_fp16_interpreter.allocate_tensors()
            tflite_fp16_available = True
            print("[+] TFLite FP16 model loaded successfully")
        else:
            print("[!] TFLite FP16 model not found")
            tflite_fp16_available = False
    except Exception as e:
        print(f"[!] Error loading TFLite FP16: {e}")
        tflite_fp16_available = False

# Load TFLite models on startup
load_tflite_fp16_model()


# ========================
#  DATABASE CONNECT
# ========================
def get_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "presensi"),
        port=int(os.getenv("DB_PORT", "3306"))
    )

# ========================
#  MODEL INFERENCE HELPERS
# ========================

def extract_embedding_deepface(img_path):
    """Extract embedding menggunakan DeepFace original"""
    try:
        rep = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False
        )
        if len(rep) > 0:
            return np.array(rep[0]["embedding"])
        return None
    except Exception as e:
        print(f"[!] DeepFace extraction error: {e}")
        return None

def extract_embedding_tflite_fp16(img_path):
    """Extract embedding menggunakan TFLite FP16 quantized model"""
    try:
        img = cv2.imread(img_path) if isinstance(img_path, str) else img_path
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        input_details = tflite_fp16_interpreter.get_input_details()
        output_details = tflite_fp16_interpreter.get_output_details()
        
        img_batch = np.expand_dims(img, axis=0)
        tflite_fp16_interpreter.set_tensor(input_details[0]['index'], img_batch)
        tflite_fp16_interpreter.invoke()
        
        embedding = tflite_fp16_interpreter.get_tensor(output_details[0]['index'])[0]
        return embedding
    except Exception as e:
        print(f"[!] TFLite FP16 extraction error: {e}")
        return None

# ========================
#  HALAMAN ADMIN REGISTER
# ========================
@app.route("/")
def index():
    return render_template("presensi.html")

@app.route("/test-camera")
def test_camera():
    return render_template("test_camera.html")

@app.route("/admin")
def admin_page():
    return render_template("admin_register.html")


@app.route("/admin/register", methods=["POST"])
def admin_register():

    name = request.form["name"]
    photo = request.files["photo"]
    model_type = request.form.get("model_type", "tflite_fp16")  # deepface or tflite_fp16

    filename = name.replace(" ", "_") + ".jpg"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    photo.save(path)

    # Ekstraksi embedding berdasarkan model type
    try:
        if model_type == "tflite_fp16" and tflite_fp16_available:
            rep = extract_embedding_tflite_fp16(path)
        else:
            rep = extract_embedding_deepface(path)
        
        if rep is None or len(rep) == 0:
            return f"Error deteksi wajah! Wajah tidak terdeteksi dengan model {model_type}."
        
        rep = np.array(rep)
    except Exception as e:
        return f"Error deteksi wajah! <br>Detail: {e}"

    # Simpan embedding sebagai BLOB base64
    emb_blob = base64.b64encode(pickle.dumps(rep)).decode('utf-8')

    db = get_db()
    cursor = db.cursor()
    sql = "INSERT INTO users (name, photo, embedding) VALUES (%s, %s, %s)"
    cursor.execute(sql, (name, filename, emb_blob))
    db.commit()

    return f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Registrasi Berhasil</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" />
        <style>
            body {{
                background: #F0F4F8;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            }}
            .success-container {{
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
                padding: 40px;
                max-width: 500px;
                text-align: center;
            }}
            .success-icon {{
                font-size: 60px;
                color: #28a745;
                margin-bottom: 20px;
            }}
            .success-title {{
                color: #0066CC;
                font-weight: 700;
                font-size: 28px;
                margin-bottom: 15px;
            }}
            .photo-preview {{
                margin: 30px 0;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }}
            .photo-preview img {{
                width: 100%;
                max-width: 300px;
                display: block;
                margin: 0 auto;
            }}
            .user-info {{
                background: #F8F9FA;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #0066CC;
            }}
            .user-info label {{
                font-weight: 600;
                color: #666;
                font-size: 12px;
                text-transform: uppercase;
                display: block;
                margin-bottom: 5px;
            }}
            .user-info p {{
                font-size: 18px;
                color: #333;
                margin: 0;
            }}
            .button-group {{
                margin-top: 30px;
                display: flex;
                gap: 10px;
                justify-content: center;
            }}
            .btn-back {{
                background: #0066CC;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
            }}
            .btn-back:hover {{
                background: #0052A3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 102, 204, 0.3);
                color: white;
            }}
            .btn-next {{
                background: #00AA66;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
            }}
            .btn-next:hover {{
                background: #008A52;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 170, 102, 0.3);
                color: white;
            }}
            .success-message {{
                color: #28a745;
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="success-container">
            <div class="success-icon">
                <i class="bi bi-check-circle-fill"></i>
            </div>
            
            <div class="success-message">
                <i class="bi bi-check-lg"></i> Registrasi Berhasil!
            </div>
            
            <h1 class="success-title">Karyawan Terdaftar</h1>
            
            <div class="user-info">
                <label>Nama Karyawan</label>
                <p>{name}</p>
            </div>
            
            <div class="photo-preview">
                <img src="/static/uploads/{filename}" alt="Foto {name}">
            </div>
            
            <p style="color: #666; font-size: 14px; margin: 20px 0;">
                <i class="bi bi-info-circle"></i>
                Wajah karyawan telah berhasil didaftarkan dan disimpan ke database.
            </p>
            
            <div class="button-group">
                <a href="/admin" class="btn-back">
                    <i class="bi bi-arrow-left"></i> Daftar Lagi
                </a>
                <a href="/presensi-user" class="btn-next">
                    <i class="bi bi-arrow-right"></i> Presensi
                </a>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """


# ========================
#  HALAMAN PRESENSI (USER)
# ========================
@app.route("/presensi-user")
def presensi_user():
    return render_template("presensi.html")  # ada ambil kamera + upload foto


# ========================
#  FUNGSI PEMBANDING ARC FACE
# ========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ========================
#  FACE DETECTION HELPERS
# ========================
def detect_face_with_bbox(img):
    """
    Detect face dan return koordinat bounding box
    Menggunakan OpenCV Cascade atau DeepFace detector
    
    Returns:
        List of (x, y, w, h) tuples
    """
    try:
        # Pakai DeepFace detector (RetinaFace)
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="opencv",  # Bisa ganti ke "retinaface" untuk lebih akurat
            enforce_detection=False,
            align=False
        )
        
        face_coords = []
        for face_dict in faces:
            # face_dict contains {'facial_area': {...}, 'confidence': ...}
            facial_area = face_dict.get('facial_area', {})
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)
            
            if w > 0 and h > 0:
                face_coords.append((x, y, w, h))
        
        return face_coords
    
    except Exception as e:
        print(f"[!] Face detection error: {e}")
        # Fallback ke OpenCV Cascade
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            return [(x, y, w, h) for (x, y, w, h) in faces]
        except:
            return []


def draw_bounding_boxes(img, face_coords, face_info=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes pada gambar dengan label berdasarkan face_info
    
    Args:
        img: OpenCV image
        face_coords: List of (x, y, w, h) tuples
        face_info: List of dicts dengan info masing-masing face (opsional)
        color: RGB color tuple
        thickness: Line thickness
    
    Returns:
        Image dengan bounding boxes
    """
    img_copy = img.copy()
    
    for idx, (x, y, w, h) in enumerate(face_coords):
        # Draw rectangle
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        
        # Draw filled rectangle untuk background text
        label_height = 25
        cv2.rectangle(img_copy, (x, y - label_height), (x + 150, y), color, -1)
        
        # Buat label berdasarkan face_info atau hanya nomor urut
        if face_info and idx < len(face_info):
            label = f"Face {idx + 1}: {face_info[idx].get('name', '?')}"
        else:
            label = f"Face {idx + 1}"
        
        # Put text
        cv2.putText(img_copy, label, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy


def extract_embedding_from_face_area(img, x, y, w, h, model_type="deepface"):
    """
    Extract embedding dari area wajah spesifik
    
    Args:
        img: OpenCV image (numpy array)
        x, y, w, h: Koordinat dan ukuran face area
        model_type: Tipe model yang digunakan
    
    Returns:
        embedding array atau None
    """
    try:
        # Crop face area
        face_area = img[y:y+h, x:x+w]
        
        if model_type == "tflite_fp16" and tflite_fp16_available:
            return extract_embedding_tflite_fp16(face_area)
        else:
            return extract_embedding_deepface(face_area)
    except Exception as e:
        print(f"[!] Error extracting embedding from face area: {e}")
        return None


# ========================
#  PRESENSI VIA KAMERA (BASE64)
# ========================
@app.route("/presensi-kamera", methods=["POST"])
def presensi_kamera():

    try:
        image_data = request.form["image_data"]
        model_type = request.form.get("model_type", "tflite_fp16")  # deepface or tflite_fp16
        
        image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces
        face_coords = detect_face_with_bbox(img)
        
        if not face_coords:
            return jsonify({
                "status": False, 
                "message": "Wajah tidak terdeteksi!",
                "image_with_bbox": None,
                "results": []
            })

        # Ambil semua users dari DB sekali saja
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        db_users = cursor.fetchall()

        # Process setiap wajah yang terdeteksi
        face_results = []
        
        for idx, (x, y, w, h) in enumerate(face_coords):
            # Extract embedding untuk face area ini
            user_embed = extract_embedding_from_face_area(img, x, y, w, h, model_type)
            
            if user_embed is None:
                face_results.append({
                    "face_num": idx + 1,
                    "status": False,
                    "message": "Wajah tidak terdeteksi!",
                    "name": "Unknown",
                    "score": 0.0
                })
                continue
            
            user_embed = np.array(user_embed)
            best_user = None
            best_score = -1
            
            # Cari user yang paling cocok
            for row in db_users:
                emb_db = pickle.loads(base64.b64decode(row["embedding"]))
                emb_db = np.array(emb_db)
                sim = cosine_similarity(user_embed, emb_db)
                
                if sim > best_score:
                    best_score = sim
                    best_user = row
            
            # Cek threshold recognition
            if best_score < 0.40:
                face_results.append({
                    "face_num": idx + 1,
                    "status": False,
                    "message": "Wajah tidak dikenali!",
                    "name": "Unknown",
                    "score": float(best_score)
                })
            else:
                # Catat absensi jika score bagus
                cursor.execute("INSERT INTO absensi (user_id, waktu) VALUES (%s, NOW())",
                               (best_user["id"],))
                db.commit()
                
                face_results.append({
                    "face_num": idx + 1,
                    "status": True,
                    "message": f"Presensi Berhasil",
                    "name": best_user["name"],
                    "score": float(best_score)
                })
        
        # Draw bounding boxes dengan info dari hasil recognition
        img_with_bbox = draw_bounding_boxes(img, face_coords, face_info=face_results, color=(0, 255, 0), thickness=3)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', img_with_bbox)
        img_bbox_b64 = base64.b64encode(buffer).decode('utf-8')

        # Cek apakah ada yang berhasil presensi
        success_count = sum(1 for r in face_results if r["status"])
        
        if success_count > 0:
            message = f"Presensi Berhasil: {success_count} dari {len(face_results)} wajah"
        else:
            message = f"Tidak ada wajah yang dikenali ({len(face_results)} wajah terdeteksi)"

        return jsonify({
            "status": success_count > 0,
            "message": message,
            "image_with_bbox": f"data:image/jpeg;base64,{img_bbox_b64}",
            "model": model_type,
            "results": face_results,
            "total_faces": len(face_results)
        })

    except Exception as e:
        return jsonify({"status": False, "message": f"Error: {str(e)}", "results": []})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)

