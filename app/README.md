# Flower Recognition MVP (Oxford 102 Category Flower Dataset)

MVP web application for flower species identification based on the CNN architecture (ResNet-50 transfer learning & custom CNN) defined in `Proyecto_Flores_Oxford102_v2.ipynb`.

---

## 📁 Project Structure

```text
├── Proyecto_Flores_Oxford102_v2.ipynb   # Main Jupyter Notebook
├── app/                                 # Web Application Root
│   ├── app.py                           # Flask backend API & inference runner
│   ├── requirements.txt                 # Python dependencies
│   ├── Procfile                         # Process file for Render / Gunicorn
│   ├── render.yaml                      # Render Blueprint deployment configuration
│   ├── README.md                        # Documentation & setup guide
│   ├── model/                           # Directory for model checkpoints
│   │   ├── model_b_best.pt              # PyTorch weights (from Notebook)
│   │   └── flower_model_b.onnx          # Optional ONNX export
│   └── templates/
│       └── index.html                   # Responsive frontend interface
```

---

## ⚙️ Model Setup

The app automatically handles inference for the **102 Oxford flower species**:
1. It first attempts to load an ONNX model from `model/flower_model_b.onnx`.
2. If not present, it loads the PyTorch weights checkpoint from `model/model_b_best.pt`.
3. If neither file exists yet in the folder, it initializes the architecture with ImageNet feature layers as a working pipeline demonstration.

### Copying your trained weights from the notebook:
After training `model_b` (or exporting to ONNX) in your notebook:
- Copy `checkpoints/model_b_best.pt` to `app/model/model_b_best.pt`
- Or copy `checkpoints/flower_model_b.onnx` to `app/model/flower_model_b.onnx`

---

## 💻 Local Execution

1. Navigate to the `app` folder:
   ```bash
   cd app
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   python app.py
   ```

5. Open your browser and go to `http://localhost:5000`.

---

## 🚀 Deployment to Render.com

Render offers free hosting for web services. Follow these steps:

### Method 1: Git Repository (Recommended)

1. Push this project to GitHub or GitLab.
2. Log in to [Render.com](https://render.com).
3. Click **New +** and choose **Web Service**.
4. Connect your GitHub repository.
5. Configure the service settings:
   - **Name**: `flower-recognition-oxford102` (or your choice)
   - **Root Directory**: `app`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type**: `Free`
6. Click **Create Web Service**.

### Method 2: Render Blueprint (`render.yaml`)

1. Connect your repository to Render.
2. Select **Blueprints** on Render and choose your repository.
3. Render will automatically read `render.yaml` and configure the service.

