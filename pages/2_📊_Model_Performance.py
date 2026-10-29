"""Model status and saved evaluation results; no generated performance data."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from src.config import COLORS, PROJECT_TITLE
from src.session_state import init_state
from src.theme import LIGHT_THEME_CSS
from src.anomaly_detector import CHECKPOINT_PATH, _load_model
from src.calibration import checkpoint_digest, SCORING_VERSION
from src.performance import evaluation_rows, numeric_csv, roc_points
from src.evaluate import compute_confusion_matrix, compute_classification_metrics, compute_roc_auc

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title=f'Model Performance · {PROJECT_TITLE}', page_icon='📊', layout='wide')
init_state()
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
st.title('📊 Model Performance')
st.markdown('Evaluation metrics and training curves for the VAE model.')
if CHECKPOINT_PATH.is_file():
    try:
        _load_model(CHECKPOINT_PATH)
        st.success(f'Checkpoint loaded successfully: {CHECKPOINT_PATH.name}')
        st.caption('Loading weights verifies compatibility, not training quality or defect accuracy.')
    except Exception as exc:
        st.error(f'Checkpoint cannot be loaded: {exc}')
else:
    st.warning(f'Checkpoint missing: {CHECKPOINT_PATH}. Historical logs do not prove the current model is available.')

rows = None
try:
    rows = evaluation_rows(ROOT / 'outputs/evaluation_results.csv')
    metadata_path = ROOT / 'outputs/evaluation_results.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        matched = (CHECKPOINT_PATH.is_file() and
                   metadata['checkpoint_sha256'] == checkpoint_digest(CHECKPOINT_PATH) and
                   metadata['scoring_version'] == SCORING_VERSION)
        st.info('Saved evaluation matches the installed checkpoint and scoring method.' if matched else
                'Historical evaluation: checkpoint or scoring method differs from the installed model.')
    else:
        st.warning('Historical evaluation CSV: checkpoint and scoring provenance are unavailable. These are saved results, not a new evaluation of the current model.')
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f'Evaluation unavailable: {exc}')
    rows = None

st.divider()
st.subheader('📋 Classification Metrics')
metrics = {}
if rows:
    labels = np.array([r['true_label'] for r in rows])
    scores = np.array([r['anomaly_score'] for r in rows])
    predictions = np.array([r['pred_label'] for r in rows])
    cm = compute_confusion_matrix(labels, predictions)
    metrics = compute_classification_metrics(**cm)
    st.caption(f"{len(rows)} saved test images; good = normal, other categories = defect. Threshold(s): {sorted(set(r['threshold'] for r in rows))}. Undefined precision/recall/F1 denominators use zero.")
for col, name, key in zip(st.columns(4), ['Accuracy','Precision','Recall','F1 Score'], ['accuracy','precision','recall','f1_score']):
    col.metric(name, f'{metrics[key]:.4f}' if metrics else 'N/A')
if rows:
    st.write('Confusion matrix — rows: actual Normal, Anomaly; columns: predicted Normal, Anomaly')
    st.dataframe({'Actual': ['Normal', 'Anomaly'], 'Normal': [cm['tn'], cm['fn']], 'Anomaly': [cm['fp'], cm['tp']]}, hide_index=True)
    if all('mse' in r and 'ssim' in r for r in rows):
        try:
            quality = numeric_csv(ROOT / 'outputs/evaluation_results.csv', ['mse', 'ssim'])
            q1, q2 = st.columns(2)
            q1.metric('Mean reconstruction MSE', f"{np.mean([r['mse'] for r in quality]):.6f}")
            q2.metric('Mean reconstruction SSIM', f"{np.mean([r['ssim'] for r in quality]):.4f}")
        except ValueError as exc:
            st.warning(f'Reconstruction metrics unavailable: {exc}')
    else:
        st.info('MSE and SSIM were not saved in this historical evaluation. Run evaluation with the checkpoint and dataset to calculate them.')

st.divider()
left, right = st.columns(2)
with left:
    st.subheader('ROC Curve')
    if rows:
        try:
            fpr, tpr = roc_points(labels, scores)
            auc = compute_roc_auc(labels, scores)
            st.metric('AUROC', f'{auc:.4f}')
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(fpr, tpr, color=COLORS['primary'], label=f'Saved evaluation (AUC {auc:.4f})')
            ax.plot([0,1],[0,1], 'k--', alpha=.4, label='Chance reference')
            ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
            ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        except ValueError as exc:
            st.info(str(exc))
    else:
        st.info('No valid evaluation file available.')
with right:
    st.subheader('Training Loss')
    try:
        history = numeric_csv(ROOT / 'outputs/training_history.csv', ['epoch','total_loss','recon_loss','kl_loss'])
        st.caption('Saved training log; association with the installed checkpoint is unverified. No validation loss was recorded.')
        fig, ax = plt.subplots(figsize=(5,4))
        for key, label in [('total_loss','Total'),('recon_loss','Reconstruction'),('kl_loss','KL')]:
            ax.plot([r['epoch'] for r in history], [r[key] for r in history], label=label)
        ax.set(xlabel='Epoch', ylabel='Loss'); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    except (OSError, ValueError) as exc:
        st.info(f'Training history unavailable: {exc}')
st.divider()
st.subheader('🖼️ Reconstruction Samples')
samples = [r for r in (rows or []) if Path(r.get('file_path','')).is_file() and Path(r.get('reconstruction_path','')).is_file()][:4]
if samples:
    for col, row in zip(st.columns(4), samples):
        with col:
            try:
                with Image.open(row['file_path']) as original, Image.open(row['reconstruction_path']) as reconstruction:
                    st.image(original.copy(), caption=f"Original: {row['filename']}", use_container_width=True)
                    st.image(reconstruction.copy(), caption='Saved evaluation reconstruction', use_container_width=True)
            except OSError as exc:
                st.warning(f'Sample unavailable: {exc}')
else:
    st.info('No original/reconstruction pairs are available. Run evaluation with an accessible test dataset and checkpoint to save actual pairs.')
st.divider()
st.caption('Charts use saved files only. Historical results are not evidence of current model quality.')
