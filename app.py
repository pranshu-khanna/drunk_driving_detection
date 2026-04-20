import cv2
import dlib
import numpy as np
import base64
import json
import math
import time
import hashlib
import secrets
import os
import pickle
import tempfile
from collections import deque
from datetime import datetime
from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS

# ── Speech module ─────────────────────────────────────────────────────────────
import speech_analysis as sa

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app, supports_credentials=True)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
USERS_FILE  = os.path.join(DATA_DIR, 'users.json')
MODEL_DIR   = os.path.join(DATA_DIR, 'models')
TRAIN_DIR   = os.path.join(DATA_DIR, 'training_images')
MODEL_PATH  = os.path.join(MODEL_DIR, 'drunk_sober_model.pkl')
CACHE_PATH  = os.path.join(MODEL_DIR, 'training_features_cache.json')
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'drunk'),  exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'sober'),  exist_ok=True)

# ── dlib face detector + landmark predictor ───────────────────────────────────
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ── dlib 68-point landmark indices ────────────────────────────────────────────
LEFT_EYE    = list(range(42, 48))
RIGHT_EYE   = list(range(36, 42))
NOSE_TIP    = 30
CHIN        = 8
LEFT_EAR    = 0
RIGHT_EAR   = 16
LEFT_MOUTH  = 48
RIGHT_MOUTH = 54
LEFT_BROW   = 17
RIGHT_BROW  = 26
LEFT_CHEEK  = 1
RIGHT_CHEEK = 15
# Mouth landmarks: outer lip corners + top/bottom
MOUTH_TOP    = 51   # upper lip centre
MOUTH_BOTTOM = 57   # lower lip centre
MOUTH_LEFT   = 48
MOUTH_RIGHT  = 54

# ── Per-session state ─────────────────────────────────────────────────────────
user_state = {}

def get_state(username):
    if username not in user_state:
        user_state[username] = {
            'ear_history':        deque(maxlen=20),
            'sym_history':        deque(maxlen=20),
            'redness_history':    deque(maxlen=20),
            'mouth_history':      deque(maxlen=20),
            'tilt_history':       deque(maxlen=20),
            'prev_ear':           None,
            'cal_frames':         [],
            'calibrating':        False,
            'speech_cal_recordings': [],
            'speech_cal_idx':        0,
        }
    return user_state[username]

# ── JSON helpers ──────────────────────────────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

# ── dlib landmark helper ──────────────────────────────────────────────────────
def shape_to_list(shape):
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

# ══════════════════════════════════════════════════════════════════════════════
# FACIAL FEATURE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

def eye_aspect_ratio(pts, idx):
    """EAR — eye openness ratio."""
    eye = [pts[i] for i in idx]
    v1  = math.dist(eye[1], eye[5])
    v2  = math.dist(eye[2], eye[4])
    h1  = math.dist(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h1 + 1e-6)


def eye_redness_score(frame, pts, eye_idx):
    """
    Measure redness/glassiness in the eye region.
    Returns a 0-1 score: higher = redder / more bloodshot.
    Method: crop eye bounding box, compute ratio of red channel
    dominance over green in the sclera area.
    """
    eye_pts = np.array([pts[i] for i in eye_idx], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(eye_pts)
    # Add padding
    pad = max(4, int(h * 0.4))
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    # BGR → float
    roi_f = roi.astype(np.float32)
    b, g, r = roi_f[:,:,0], roi_f[:,:,1], roi_f[:,:,2]
    # Redness = how much red exceeds green (sclera redness indicator)
    red_excess = np.clip(r - g, 0, None)
    total      = r + g + b + 1e-6
    redness    = float(np.mean(red_excess / total))
    # Glassiness proxy: low local contrast in the eye region
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray_roi)) / 128.0
    glassiness = max(0.0, 1.0 - contrast)
    # Combine: 70% redness, 30% glassiness
    return min(1.0, redness * 2.5 * 0.70 + glassiness * 0.30)


def mouth_openness_ratio(pts):
    """
    Ratio of vertical mouth gap to horizontal mouth width.
    Higher = more open / slack jaw.
    """
    top    = np.array(pts[MOUTH_TOP])
    bottom = np.array(pts[MOUTH_BOTTOM])
    left   = np.array(pts[MOUTH_LEFT])
    right  = np.array(pts[MOUTH_RIGHT])
    vertical   = math.dist(top, bottom)
    horizontal = math.dist(left, right) + 1e-6
    return vertical / horizontal


def head_tilt_angle(pts):
    """
    Absolute head tilt (roll) in degrees from the ear-to-ear line.
    Pure landmark-based, no history needed.
    """
    le   = pts[LEFT_EAR]
    re   = pts[RIGHT_EAR]
    roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    return abs(roll)


def facial_symmetry_score(pts):
    nose_x = pts[NOSE_TIP][0]
    pairs  = [(LEFT_CHEEK, RIGHT_CHEEK), (LEFT_MOUTH, RIGHT_MOUTH), (LEFT_BROW, RIGHT_BROW)]
    diffs  = []
    for l, r in pairs:
        ld = abs(pts[l][0] - nose_x)
        rd = abs(pts[r][0] - nose_x)
        t  = ld + rd
        if t > 0:
            diffs.append(abs(ld - rd) / t)
    if not diffs:
        return 100.0
    return max(0.0, min(100.0, (1 - np.mean(diffs) / 0.15) * 100))


def extract_feature_vector(frame):
    """
    Extract a feature vector from a frame for ML training/inference.
    Returns (vector, landmarks_dict) or (None, None) if no face detected.
    Feature vector: [ear, eye_redness, mouth_ratio, head_tilt, symmetry]
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces) == 0:
        return None, None

    shape = predictor(gray, faces[0])
    pts   = shape_to_list(shape)

    left_ear  = eye_aspect_ratio(pts, LEFT_EYE)
    right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
    avg_ear   = (left_ear + right_ear) / 2

    left_red  = eye_redness_score(frame, pts, LEFT_EYE)
    right_red = eye_redness_score(frame, pts, RIGHT_EYE)
    avg_red   = (left_red + right_red) / 2

    mouth_ratio = mouth_openness_ratio(pts)
    tilt        = head_tilt_angle(pts)
    sym         = facial_symmetry_score(pts)

    vec = np.array([avg_ear, avg_red, mouth_ratio, tilt, sym], dtype=np.float32)
    lm  = {
        'ear': avg_ear, 'eye_redness': avg_red,
        'mouth_ratio': mouth_ratio, 'head_tilt': tilt, 'sym': sym,
        'landmarks': pts   # raw 68-point list for frontend overlay
    }
    return vec, lm


def extract_metrics(frame, st):
    """Extract per-frame metrics and update session state histories."""
    vec, lm = extract_feature_vector(frame)
    if lm is None:
        return None

    st['ear_history'].append(lm['ear'])
    st['redness_history'].append(lm['eye_redness'])
    st['mouth_history'].append(lm['mouth_ratio'])
    st['tilt_history'].append(lm['head_tilt'])
    st['sym_history'].append(lm['sym'])

    return lm


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM ML MODEL  (sklearn RandomForest, stored as pickle)
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def save_model(model):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)


def collect_training_data():
    """
    Walk TRAIN_DIR/drunk and TRAIN_DIR/sober, extract feature vectors
    from each image file.  Returns X (n_samples, 5), y (n_samples,).
    """
    from sklearn.preprocessing import StandardScaler  # local import — lazy

    X, y = [], []
    label_map = {'drunk': 1, 'sober': 0}

    for label_name, label_val in label_map.items():
        folder = os.path.join(TRAIN_DIR, label_name)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        for fname in files:
            fpath = os.path.join(folder, fname)
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            vec, _ = extract_feature_vector(frame)
            if vec is not None:
                X.append(vec)
                y.append(label_val)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_model():
    """Train a RandomForest on collected images. Returns (model, metrics_dict)."""
    from sklearn.ensemble          import RandomForestClassifier
    from sklearn.model_selection   import cross_val_score
    from sklearn.preprocessing     import StandardScaler
    from sklearn.pipeline          import Pipeline

    X, y = collect_training_data()
    if len(X) < 4:
        raise ValueError(f'Need at least 4 labelled images (got {len(X)}). '
                         'Upload more to data/training_images/drunk and /sober.')

    n_drunk = int(np.sum(y == 1))
    n_sober = int(np.sum(y == 0))
    if n_drunk == 0 or n_sober == 0:
        raise ValueError('Need at least one image in BOTH drunk and sober folders.')

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(n_estimators=200, max_depth=6,
                                          random_state=42, class_weight='balanced')),
    ])

    # Cross-val accuracy (only if enough samples)
    cv_scores = []
    if len(X) >= 6:
        cv_scores = cross_val_score(pipe, X, y, cv=min(5, len(X) // 2), scoring='accuracy')

    pipe.fit(X, y)
    save_model(pipe)

    # ── Save feature cache so export_data never needs to re-run dlib ─────────
    feature_names = ['ear', 'eye_redness', 'mouth_ratio', 'head_tilt', 'symmetry']
    cache_records = []
    for vec, lbl in zip(X, y):
        row = {name: round(float(val), 4) for name, val in zip(feature_names, vec)}
        row['label'] = 'drunk' if lbl == 1 else 'sober'
        cache_records.append(row)
    with open(CACHE_PATH, 'w') as f:
        json.dump({'data': cache_records, 'total': len(cache_records),
                   'created_at': datetime.now().isoformat()}, f, indent=2)

    importances = pipe.named_steps['clf'].feature_importances_

    return pipe, {
        'n_drunk':      n_drunk,
        'n_sober':      n_sober,
        'total_images': len(X),
        'cv_accuracy':  round(float(np.mean(cv_scores)) * 100, 1) if len(cv_scores) else None,
        'feature_importance': {k: round(float(v), 3)
                               for k, v in zip(feature_names, importances)},
    }


def model_predict(model, lm_dict):
    """
    Run the trained model on a landmarks dict.
    Returns {'label': 'drunk'|'sober', 'confidence': 0-100, 'prob_drunk': 0-1}
    """
    vec = np.array([[lm_dict['ear'], lm_dict['eye_redness'],
                     lm_dict['mouth_ratio'], lm_dict['head_tilt'],
                     lm_dict['sym']]], dtype=np.float32)
    proba     = model.predict_proba(vec)[0]
    # classes_ order may vary — find which index is label 1 (drunk)
    classes   = list(model.named_steps['clf'].classes_)
    drunk_idx = classes.index(1) if 1 in classes else 0
    prob_drunk = float(proba[drunk_idx])
    label      = 'drunk' if prob_drunk >= 0.5 else 'sober'
    confidence = prob_drunk * 100 if label == 'drunk' else (1 - prob_drunk) * 100
    return {'label': label, 'confidence': round(confidence, 1), 'prob_drunk': round(prob_drunk, 3)}


# ── Scoring (rule-based fallback when no trained model exists) ────────────────
def compute_scores(lm, st, baseline, speech_score=None, ml_result=None):
    ear         = lm['ear']
    eye_redness = lm['eye_redness']
    mouth_ratio = lm['mouth_ratio']
    head_tilt   = lm['head_tilt']
    sym         = lm['sym']

    mean_ear     = np.mean(st['ear_history'])    if st['ear_history']    else ear
    mean_red     = np.mean(st['redness_history'])if st['redness_history']else eye_redness
    mean_mouth   = np.mean(st['mouth_history'])  if st['mouth_history']  else mouth_ratio
    mean_tilt    = np.mean(st['tilt_history'])   if st['tilt_history']   else head_tilt
    mean_sym     = np.mean(st['sym_history'])    if st['sym_history']    else sym
    ear_std      = np.std(st['ear_history'])     if len(st['ear_history']) > 5 else 0

    if baseline:
        b_ear   = baseline['ear']
        b_red   = baseline.get('eye_redness', 0.05)
        b_mouth = baseline.get('mouth_ratio', 0.10)
        b_tilt  = baseline.get('head_tilt', 2.0)
        b_sym   = baseline['sym']

        ear_drop    = max(0, b_ear - mean_ear)
        droop_score = min(100, (ear_drop / max(0.05, b_ear * 0.4)) * 100)
        droop_score = min(100, droop_score + max(0, ear_std - 0.01) * 300)

        redness_score = min(100, max(0, (mean_red - b_red) / max(0.05, b_red + 0.1)) * 100)

        mouth_score = min(100, max(0, (mean_mouth - b_mouth) / max(0.05, b_mouth + 0.1)) * 100)

        tilt_score = min(100, max(0, (mean_tilt - b_tilt) / max(3, b_tilt + 5)) * 100)

        sym_score = min(100, (max(0, b_sym - mean_sym) / max(5, b_sym * 0.3)) * 100)
    else:
        droop_score   = min(100, max(0, (0.30 - mean_ear) / 0.15) * 100 + ear_std * 200)
        redness_score = min(100, mean_red * 300)
        mouth_score   = min(100, max(0, (mean_mouth - 0.08) / 0.12) * 100)
        tilt_score    = min(100, (mean_tilt / 15) * 100)
        sym_score     = max(0, 100 - mean_sym)

    # ── If ML model result available, blend it in ─────────────────────────────
    if ml_result is not None:
        ml_score = ml_result['prob_drunk'] * 100
        rule_overall = (droop_score   * 0.30 +
                        redness_score * 0.25 +
                        mouth_score   * 0.20 +
                        tilt_score    * 0.15 +
                        sym_score     * 0.10)
        facial_overall = rule_overall * 0.40 + ml_score * 0.60
    else:
        facial_overall = (droop_score   * 0.30 +
                          redness_score * 0.25 +
                          mouth_score   * 0.20 +
                          tilt_score    * 0.15 +
                          sym_score     * 0.10)

    # ── Blend with speech score if present ────────────────────────────────────
    if speech_score is not None:
        overall = facial_overall * 0.70 + speech_score * 0.30
    else:
        overall = facial_overall

    result = {
        'detected':      True,
        'overall':       round(overall, 1),
        'has_baseline':  baseline is not None,
        'ml_prediction': ml_result,
        'signals': {
            'eye_droopiness': {
                'score':        round(droop_score, 1),
                'raw_ear':      round(ear, 3),
                'mean_ear':     round(mean_ear, 3),
                'baseline_ear': round(baseline['ear'], 3) if baseline else None,
                'label':        'Eye Droopiness',
            },
            'eye_redness': {
                'score':           round(redness_score, 1),
                'raw_redness':     round(eye_redness, 3),
                'mean_redness':    round(mean_red, 3),
                'baseline_redness': round(baseline.get('eye_redness', 0), 3) if baseline else None,
                'label':           'Eye Redness / Glassiness',
            },
            'mouth_openness': {
                'score':          round(mouth_score, 1),
                'raw_ratio':      round(mouth_ratio, 3),
                'mean_ratio':     round(mean_mouth, 3),
                'baseline_ratio': round(baseline.get('mouth_ratio', 0), 3) if baseline else None,
                'label':          'Mouth Openness (Slack Jaw)',
            },
            'head_tilt': {
                'score':          round(tilt_score, 1),
                'raw_tilt_deg':   round(head_tilt, 1),
                'mean_tilt_deg':  round(mean_tilt, 1),
                'baseline_tilt':  round(baseline.get('head_tilt', 0), 1) if baseline else None,
                'label':          'Head Tilt Angle',
            },
            'facial_symmetry': {
                'score':        round(sym_score, 1),
                'symmetry_pct': round(mean_sym, 1),
                'baseline_sym': round(baseline['sym'], 1) if baseline else None,
                'label':        'Facial Symmetry',
            },
        },
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/register', methods=['POST'])
def register():
    data  = request.get_json()
    uname = data.get('username', '').strip().lower()
    pwd   = data.get('password', '')
    if not uname or not pwd:
        return jsonify({'error': 'Username and password required'}), 400
    if len(uname) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if len(pwd) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    users = load_users()
    if uname in users:
        return jsonify({'error': 'Username already taken'}), 409
    users[uname] = {
        'username':   uname,
        'password':   hash_password(pwd),
        'created_at': datetime.now().isoformat(),
        'baseline':   None,
    }
    save_users(users)
    session['username'] = uname
    return jsonify({'success': True, 'username': uname, 'has_baseline': False, 'redirect': '/setup'})


@app.route('/api/login', methods=['POST'])
def login():
    data  = request.get_json()
    uname = data.get('username', '').strip().lower()
    pwd   = data.get('password', '')
    users = load_users()
    user  = users.get(uname)
    if not user or user['password'] != hash_password(pwd):
        return jsonify({'error': 'Invalid username or password'}), 401
    session['username'] = uname
    has_bl = user['baseline'] is not None
    return jsonify({
        'success': True, 'username': uname,
        'has_baseline':  has_bl,
        'baseline_date': user['baseline']['created_at'] if user['baseline'] else None,
        'redirect': '/' if has_bl else '/setup',
    })


@app.route('/api/logout', methods=['POST'])
def logout():
    u = session.get('username')
    if u and u in user_state: del user_state[u]
    session.clear()
    return jsonify({'success': True})


@app.route('/api/me')
def me():
    uname = session.get('username')
    if not uname: return jsonify({'logged_in': False})
    users = load_users()
    user  = users.get(uname)
    if not user: return jsonify({'logged_in': False})
    return jsonify({
        'logged_in': True, 'username': uname,
        'has_baseline':  user['baseline'] is not None,
        'baseline_date': user['baseline']['created_at'] if user['baseline'] else None,
        'baseline':      user['baseline'],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# FACIAL CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/calibrate/start', methods=['POST'])
def calibrate_start():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    st = get_state(uname)
    st['cal_frames'] = [];  st['calibrating'] = True
    return jsonify({'success': True})


@app.route('/api/calibrate/frame', methods=['POST'])
def calibrate_frame():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    st = get_state(uname)
    if not st['calibrating']: return jsonify({'error': 'Not calibrating'}), 400
    data = request.get_json()
    try:
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        frame    = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    m = extract_metrics(frame, st)
    if m is None: return jsonify({'detected': False, 'frames_collected': len(st['cal_frames'])})
    st['cal_frames'].append(m)
    return jsonify({'detected': True, 'frames_collected': len(st['cal_frames'])})


@app.route('/api/calibrate/finish', methods=['POST'])
def calibrate_finish():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    st = get_state(uname);  frames = st['cal_frames']
    if len(frames) < 10:
        return jsonify({'error': f'Only {len(frames)} frames — keep face visible'}), 400

    baseline = {
        'ear':         float(np.mean([f['ear']          for f in frames])),
        'ear_std':     float(np.std( [f['ear']          for f in frames])),
        'eye_redness': float(np.mean([f['eye_redness']  for f in frames])),
        'mouth_ratio': float(np.mean([f['mouth_ratio']  for f in frames])),
        'head_tilt':   float(np.mean([f['head_tilt']    for f in frames])),
        'sym':         float(np.mean([f['sym']          for f in frames])),
        'frames':      len(frames),
        'created_at':  datetime.now().isoformat(),
        'speech':      None,
    }
    users = load_users()
    users[uname]['baseline'] = baseline
    save_users(users)
    st['calibrating'] = False;  st['cal_frames'] = []
    return jsonify({'success': True, 'baseline': {
        'ear':         round(baseline['ear'], 3),
        'eye_redness': round(baseline['eye_redness'], 3),
        'mouth_ratio': round(baseline['mouth_ratio'], 3),
        'head_tilt':   round(baseline['head_tilt'], 1),
        'sym':         round(baseline['sym'], 1),
        'frames':      baseline['frames'],
        'created_at':  baseline['created_at'],
    }})


# ═══════════════════════════════════════════════════════════════════════════════
# ML TRAINING ROUTES  —  /api/train/*
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/train/status', methods=['GET'])
def train_status():
    """Return counts of labelled images and whether a model exists."""
    drunk_dir = os.path.join(TRAIN_DIR, 'drunk')
    sober_dir = os.path.join(TRAIN_DIR, 'sober')
    exts      = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    n_drunk = len([f for f in os.listdir(drunk_dir) if f.lower().endswith(exts)])
    n_sober = len([f for f in os.listdir(sober_dir) if f.lower().endswith(exts)])
    model_exists = os.path.exists(MODEL_PATH)
    model_mtime  = None
    if model_exists:
        model_mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()

    return jsonify({
        'n_drunk':      n_drunk,
        'n_sober':      n_sober,
        'model_exists': model_exists,
        'model_trained_at': model_mtime,
        'train_dir':    TRAIN_DIR,
    })


@app.route('/api/train/upload', methods=['POST'])
def train_upload():
    """
    Upload a labelled image for training.
    Accepts JSON: { "image": "<base64 data-URI>", "label": "drunk"|"sober" }
    Or multipart form: image file + label field.
    """
    label = None
    img_bytes = None

    content_type = request.content_type or ''

    if 'multipart' in content_type:
        label = request.form.get('label', '').lower()
        f     = request.files.get('image')
        if f is None:
            return jsonify({'error': 'No image file provided'}), 400
        img_bytes = f.read()
    else:
        data  = request.get_json()
        label = (data.get('label') or '').lower()
        b64   = data.get('image', '')
        try:
            raw       = b64.split(',')[1] if ',' in b64 else b64
            img_bytes = base64.b64decode(raw)
        except Exception as e:
            return jsonify({'error': f'Image decode error: {e}'}), 400

    if label not in ('drunk', 'sober'):
        return jsonify({'error': "label must be 'drunk' or 'sober'"}), 400

    # Quick sanity check — must detect a face
    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return jsonify({'error': 'No face detected in image — please use a clear frontal face photo'}), 422

    # Save with unique name
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'{ts}.jpg'
    save_path = os.path.join(TRAIN_DIR, label, filename)
    cv2.imwrite(save_path, frame)

    return jsonify({'success': True, 'label': label, 'filename': filename})


@app.route('/api/train/run', methods=['POST'])
def train_run():
    """
    Trigger model training on all uploaded images.
    Returns accuracy metrics.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier  # ensure sklearn installed
    except ImportError:
        return jsonify({'error': 'scikit-learn not installed. Run: pip install scikit-learn'}), 503

    try:
        model, metrics = train_model()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Training failed: {e}'}), 500

    return jsonify({'success': True, 'metrics': metrics,
                    'model_path': MODEL_PATH,
                    'trained_at': datetime.now().isoformat()})


@app.route('/api/train/clear', methods=['POST'])
def train_clear():
    """
    Clear all training images and/or the trained model.
    JSON body: { "clear_images": true, "clear_model": true }
    """
    data          = request.get_json() or {}
    clear_images  = data.get('clear_images', False)
    clear_model   = data.get('clear_model', False)
    removed_images = 0

    if clear_images:
        for label in ('drunk', 'sober'):
            folder = os.path.join(TRAIN_DIR, label)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    os.remove(os.path.join(folder, fname))
                    removed_images += 1

    if clear_model and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    return jsonify({'success': True, 'removed_images': removed_images,
                    'model_cleared': clear_model})


# ═══════════════════════════════════════════════════════════════════════════════
# SPEECH CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/speech/sentences')
def speech_sentences():
    return jsonify({'sentences': sa.SPEECH_SENTENCES})


@app.route('/api/speech/availability')
def speech_availability():
    return jsonify(sa.check_availability())


@app.route('/api/speech/calibrate/start', methods=['POST'])
def speech_calibrate_start():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    st = get_state(uname)
    st['speech_cal_recordings'] = []
    st['speech_cal_idx']        = 0
    return jsonify({'success': True, 'total_sentences': len(sa.SPEECH_SENTENCES),
                    'first_sentence': sa.SPEECH_SENTENCES[0]})


@app.route('/api/speech/calibrate/record', methods=['POST'])
def speech_calibrate_record():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    if not sa.check_availability()['ready']:
        return jsonify({'error': 'whisper/librosa not installed'}), 503

    st  = get_state(uname)
    idx = st['speech_cal_idx']
    if idx >= len(sa.SPEECH_SENTENCES):
        return jsonify({'error': 'All sentences already recorded'}), 400

    data = request.get_json()
    try:
        audio_b64   = data['audio']
        audio_bytes = base64.b64decode(audio_b64.split(',')[1] if ',' in audio_b64 else audio_b64)
    except Exception as e:
        return jsonify({'error': f'Audio decode error: {e}'}), 400

    suffix = '.webm' if audio_b64.startswith('data:audio/webm') else '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        features = sa.compute_speech_features(tmp_path, sa.SPEECH_SENTENCES[idx])
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({'error': f'Analysis failed: {e}'}), 500
    finally:
        try: os.unlink(tmp_path)
        except: pass

    st['speech_cal_recordings'].append({'sentence_idx': idx, 'features': features})
    st['speech_cal_idx'] += 1
    next_idx = st['speech_cal_idx']

    return jsonify({
        'success':       True,
        'sentence_done': idx,
        'transcription': features.get('transcription', ''),
        'wpm':           round(features.get('wpm', 0), 1),
        'next_sentence': sa.SPEECH_SENTENCES[next_idx] if next_idx < len(sa.SPEECH_SENTENCES) else None,
        'done':          next_idx >= len(sa.SPEECH_SENTENCES),
    })


@app.route('/api/speech/calibrate/finish', methods=['POST'])
def speech_calibrate_finish():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    st   = get_state(uname)
    recs = st['speech_cal_recordings']
    if len(recs) < len(sa.SPEECH_SENTENCES):
        return jsonify({'error': f'Only {len(recs)}/{len(sa.SPEECH_SENTENCES)} sentences recorded'}), 400

    speech_baseline = sa.build_speech_baseline(recs)
    speech_baseline['created_at'] = datetime.now().isoformat()

    users = load_users()
    if uname in users:
        if users[uname]['baseline'] is None:
            users[uname]['baseline'] = {'speech': speech_baseline, 'created_at': datetime.now().isoformat()}
        else:
            users[uname]['baseline']['speech'] = speech_baseline
        save_users(users)

    st['speech_cal_recordings'] = [];  st['speech_cal_idx'] = 0
    return jsonify({'success': True, 'speech_baseline': {
        'wpm':         round(speech_baseline['wpm'], 1),
        'pause_ratio': round(speech_baseline['pause_ratio'], 3),
        'created_at':  speech_baseline['created_at'],
    }})


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE SPEECH TEST
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/speech/test', methods=['POST'])
def speech_test():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    if not sa.check_availability()['ready']:
        return jsonify({'error': 'whisper/librosa not installed'}), 503

    users    = load_users()
    baseline = users.get(uname, {}).get('baseline', {})
    speech_b = baseline.get('speech') if baseline else None
    if not speech_b:
        return jsonify({'error': 'No speech baseline recorded yet'}), 400

    data         = request.get_json()
    sentence_idx = data.get('sentence_idx', 0)

    try:
        audio_b64   = data['audio']
        audio_bytes = base64.b64decode(audio_b64.split(',')[1] if ',' in audio_b64 else audio_b64)
    except Exception as e:
        return jsonify({'error': f'Audio decode: {e}'}), 400

    suffix = '.webm' if audio_b64.startswith('data:audio/webm') else '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        sentence = sa.SPEECH_SENTENCES[sentence_idx % len(sa.SPEECH_SENTENCES)]
        features = sa.compute_speech_features(tmp_path, sentence)
        scores   = sa.score_speech(features, speech_b)
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({'error': f'Analysis failed: {e}'}), 500
    finally:
        try: os.unlink(tmp_path)
        except: pass

    return jsonify({'success': True, 'scores': scores, 'sentence': sentence})


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS  (facial + optional ML prediction + optional cached speech score)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/analyze', methods=['POST'])
def analyze():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    try:
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        frame    = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    st  = get_state(uname)
    lm  = extract_metrics(frame, st)
    if lm is None:
        return jsonify({'detected': False, 'message': 'No face detected'})

    users    = load_users()
    baseline = users.get(uname, {}).get('baseline')

    # Run ML model if trained — use smoothed history means, not raw single frame
    ml_result = None
    model = load_model()
    if model is not None:
        try:
            smoothed_lm = {
                'ear':         float(np.mean(st['ear_history']))     if st['ear_history']     else lm['ear'],
                'eye_redness': float(np.mean(st['redness_history'])) if st['redness_history'] else lm['eye_redness'],
                'mouth_ratio': float(np.mean(st['mouth_history']))   if st['mouth_history']   else lm['mouth_ratio'],
                'head_tilt':   float(np.mean(st['tilt_history']))    if st['tilt_history']    else lm['head_tilt'],
                'sym':         float(np.mean(st['sym_history']))     if st['sym_history']     else lm['sym'],
            }
            ml_result = model_predict(model, smoothed_lm)
        except Exception:
            ml_result = None

    speech_score = st.get('last_speech_score')
    result = compute_scores(lm, st, baseline, speech_score, ml_result)

    # Include raw landmarks for frontend overlay
    result['landmarks'] = lm.get('landmarks', [])

    if st.get('last_speech_detail'):
        result['speech'] = st['last_speech_detail']

    return jsonify(result)


@app.route('/api/speech/submit_score', methods=['POST'])
def submit_speech_score():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    st   = get_state(uname)
    st['last_speech_score']  = data.get('overall')
    st['last_speech_detail'] = data
    return jsonify({'success': True})


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE CLEAR
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/baseline/clear', methods=['POST'])
def clear_baseline():
    uname = session.get('username')
    if not uname: return jsonify({'error': 'Not logged in'}), 401
    users = load_users()
    if uname in users:
        users[uname]['baseline'] = None
        save_users(users)
    if uname in user_state:
        for k in ['ear_history', 'redness_history', 'mouth_history',
                  'tilt_history', 'sym_history']:
            user_state[uname][k].clear()
        user_state[uname]['last_speech_score']  = None
        user_state[uname]['last_speech_detail'] = None
    return jsonify({'success': True})


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT TRAINING DATA METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/train/export_data', methods=['GET'])
def export_training_data():
    """
    Export training feature vectors as JSON for graphing.
    Reads from pre-built cache (written at train time) to avoid
    re-running dlib in a Flask thread, which causes memory corruption
    ("free(): corrupted unsorted chunks").
    Run POST /api/train/run first to build the cache.
    """
    if not os.path.exists(CACHE_PATH):
        return jsonify({
            'error': (
                'Feature cache not found. '
                'Please run model training first via POST /api/train/run — '
                'this builds the cache without crashing the server.'
            )
        }), 400

    with open(CACHE_PATH, 'r') as f:
        payload = json.load(f)

    return jsonify(payload)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    uname = session.get('username')
    if uname:
        users = load_users()
        user  = users.get(uname)
        if user and user['baseline'] is None:
            from flask import redirect, url_for
            return redirect(url_for('setup'))
    return render_template('index.html')

@app.route('/setup')
def setup():
    uname = session.get('username')
    if not uname:
        from flask import redirect, url_for
        return redirect(url_for('index'))
    return render_template('setup.html')

@app.route('/profile')
def profile(): return render_template('profile.html')

@app.route('/train')
def train_page():
    return render_template('train.html')


if __name__ == '__main__':
    avail = sa.check_availability()
    print(f'🍺  SoberScan  →  http://localhost:5001')
    print(f'   Whisper:  {"✓" if avail["whisper"] else "✗ (pip install openai-whisper)"}')
    print(f'   librosa:  {"✓" if avail["librosa"] else "✗ (pip install librosa soundfile)"}')
    try:
        import sklearn
        print(f'   sklearn:  ✓  (ML model training available at /train)')
    except ImportError:
        print(f'   sklearn:  ✗ (pip install scikit-learn)')
    app.run(debug=True, host='0.0.0.0', port=5001)
