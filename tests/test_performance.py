import csv
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from src.performance import reconstruction_metrics, evaluation_rows, roc_points
from src.evaluate import evaluate
from src.model import VAE
from src.anomaly_detector import _clear_model_cache, _load_model


def test_reconstruction_metrics():
    image = np.full((32,32,3), 100, dtype=np.uint8)
    same = reconstruction_metrics(image, image)
    assert same['mse'] == 0
    assert same['ssim'] == pytest.approx(1)
    changed = reconstruction_metrics(image, np.zeros_like(image))
    assert changed['mse'] > 0
    assert changed['ssim'] < 1


def test_roc_ties_and_single_class():
    fpr, tpr = roc_points([0,1], [0.5,0.5])
    assert list(fpr) == [0,1]
    assert list(tpr) == [0,1]
    with pytest.raises(ValueError):
        roc_points([0,0], [0.1,0.2])


def test_saved_prediction_validation(tmp_path):
    path = tmp_path/'bad.csv'
    path.write_text('true_label,anomaly_score,threshold,pred_label\n1,0.8,0.5,0\n')
    with pytest.raises(ValueError, match='disagree'):
        evaluation_rows(path)


def test_evaluation_loads_selected_checkpoint_and_saves_actual_pairs(tmp_path):
    # Untrained fixture tests plumbing only; never used as product performance.
    checkpoint = tmp_path/'fixture.pth'
    model = VAE(latent_dim=4)
    torch.save({'model_state_dict':model.state_dict(), 'latent_dim':4}, checkpoint)
    for category, value in [('good', 90), ('broken', 180)]:
        folder = tmp_path/'dataset'/'test'/category
        folder.mkdir(parents=True)
        Image.fromarray(np.full((32,32,3),value,dtype=np.uint8)).save(folder/'a.png')
    output = tmp_path/'results.csv'
    result = evaluate(str(tmp_path/'dataset'), checkpoint, 0.2, output)
    assert len(result['records']) == 2
    records = evaluation_rows(output)
    assert all(Path(r['reconstruction_path']).is_file() for r in records)
    assert all(float(r['mse']) >= 0 for r in records)
    assert output.with_suffix('.json').is_file()
    other = tmp_path/'other.pth'
    torch.save({'model_state_dict':VAE(latent_dim=8).state_dict(), 'latent_dim':8}, other)
    assert _load_model(other).latent_dim == 8
    assert _load_model(checkpoint).latent_dim == 4
    _clear_model_cache()
