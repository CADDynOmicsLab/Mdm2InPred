# MDM2InPred: Prediction of MDM2 Inhibitors

MDM2InPred is a Streamlit web application for predicting the MDM2 inhibitory activity of small molecules. Given one or more SMILES strings, it computes molecular descriptors, runs them through pretrained machine learning models, and classifies each compound as a **likely MDM2 inhibitor** or **non-inhibitor**, along with a predicted **pIC₅₀** value.

## Features

- **Prediction module** — Paste SMILES strings or upload a `.smi` file and get pIC₅₀ predictions and inhibitor/non-inhibitor classification using either of two models:
  - **LightGBM**
  - **Random Forest**
- **Converter module** — Bidirectional conversion between IC₅₀ (M) and pIC₅₀ using `pIC50 = -log10(IC50)`.
- **Dataset module** — Browse and download the training, test, and external validation sets used for both models.
- **Help tab** — Embedded video walkthrough of the dashboard.
- **Contact tab** — Team and lab profile information.
- **Ask AI tab** — A chatbot (powered by Groq's OpenAI-compatible API) that answers questions about MDM2 biology and how to use the dashboard.

## Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [RDKit](https://www.rdkit.org/) — SMILES validation and cheminformatics
- [PaDELPy](https://github.com/ecrl/padelpy) — molecular descriptor/fingerprint generation (requires Java)
- [scikit-learn](https://scikit-learn.org/) / [LightGBM](https://lightgbm.readthedocs.io/) — ML models
- [joblib](https://joblib.readthedocs.io/) — model serialization
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data handling
- [OpenAI Python SDK](https://github.com/openai/openai-python) (pointed at Groq's API) — chatbot backend

## Prerequisites

- Python 3.9+
- **Java Runtime Environment (JRE)** — required by PaDELPy/PaDEL-Descriptor for descriptor calculation
- A [Groq API key](https://console.groq.com/) for the chatbot feature

## Installation

1. Clone or download this repository.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure Java is installed and available on your `PATH` (required by PaDELPy):
   ```bash
   java -version
   ```

## Configuration

The app reads a Groq API key from Streamlit secrets. Create a `.streamlit/secrets.toml` file in the project root:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

> Note: `requirements.txt` also lists `groq` and `cohere`, but the current code only calls Groq's API through the OpenAI-compatible client (`base_url="https://api.groq.com/openai/v1"`). If you don't plan to use the chatbot, this key is still required at startup since the client is initialized unconditionally.

## Required Files & Assets

The app expects the following files/folders to exist alongside the script (paths are read relative to the working directory):

| Path | Purpose |
|---|---|
| `lightgbm.pkl` | Trained LightGBM model |
| `rf.joblib` | Trained Random Forest model |
| `lightgbm_feature_importances.csv` | Feature list for the LightGBM model |
| `rf_feature_importance.csv` | Feature list for the Random Forest model |
| `main.csv` | Reference dataset |
| `3lbk.png` | MDM2 binding pocket image shown on the Home tab |
| `video.MP4` | Tutorial video shown on the Help tab |
| `images/head.jpeg`, `images/1.jpg` … `images/6.jpg` | Team/profile photos on the Contact tab |
| `static/lightgbm_train_set.csv` | LightGBM training set |
| `static/lightgbm_test_set.csv` | LightGBM test set |
| `static/predicted_activity_external_lightgbm.csv` | LightGBM external validation set |
| `static/train_set.csv` | Random Forest training set |
| `static/test_set.csv` | Random Forest test set |
| `static/validation_set.csv` | Random Forest external validation set |

These are not included in the script itself and must be provided in your deployment/project folder for all tabs to work correctly.

## Running the App

```bash
streamlit run realsmiles.py
```

By default, this opens the app at `http://localhost:8501`.

## Usage

1. **Predict tab** — Select a model (LightGBM or Random Forest), paste one or more SMILES strings (one per line) or upload a `.smi` file (up to 200 MB), then click **Run Prediction**. Results (Molecule ID, predicted pIC₅₀, and classification) are shown in a table and can be downloaded as CSV.
2. **Convert tab** — Choose a conversion direction and enter a value to convert between IC₅₀ and pIC₅₀.
3. **Dataset tab** — Click a dataset button to preview and download the corresponding training/test/validation CSV file.
4. **Help tab** — Watch the tutorial video for a walkthrough of the app.
5. **Contact tab** — View team member profiles and contact details.
6. **Ask AI tab** — Ask questions about MDM2 biology or how to use the dashboard; the assistant is scoped to the app's context and won't fabricate dataset values.

## Project Structure (expected)

```
.
├── realsmiles.py
├── requirements.txt
├── lightgbm.pkl
├── rf.joblib
├── lightgbm_feature_importances.csv
├── rf_feature_importance.csv
├── main.csv
├── 3lbk.png
├── video.MP4
├── images/
│   ├── head.jpeg
│   ├── 1.jpg ... 6.jpg
├── static/
│   ├── lightgbm_train_set.csv
│   ├── lightgbm_test_set.csv
│   ├── predicted_activity_external_lightgbm.csv
│   ├── train_set.csv
│   ├── test_set.csv
│   └── validation_set.csv
└── .streamlit/
    └── secrets.toml
```

## License

Free for **academic and non-commercial use only** © 2025–2026.

## Contact

**Dr. Sarfaraz Alam** — Assistant Professor, CADDynOmics Lab, Institute of Advanced Research, The University for Innovation, Gandhinagar
📧 sarfaraz.alam@iar.ac.in
