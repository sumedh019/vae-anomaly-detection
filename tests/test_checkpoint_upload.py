import io
import numpy as np
import pytest
import torch
from src.model import VAE
from src.anomaly_detector import detect
from src.checkpoint_upload import save_uploaded_checkpoint


def test_uploaded_weights_support_inference_and_are_session_isolated():
    stream = io.BytesIO()
    torch.save({'model_state_dict': VAE(latent_dim=4).state_dict(), 'latent_dim':4}, stream)
    data = stream.getvalue()
    first, second = {}, {}
    path = save_uploaded_checkpoint(data, first)
    assert save_uploaded_checkpoint(data, first) == path
    other = save_uploaded_checkpoint(data, second)
    assert path != other
    result = detect(np.zeros((32,32,3), dtype=np.uint8), checkpoint_path=path)
    assert result.reconstructed.shape == (32,32,3)
    with pytest.raises(RuntimeError):
        save_uploaded_checkpoint(b'not a checkpoint', first)
    assert path.is_file()
    first['_checkpoint_dir'].cleanup()
    second['_checkpoint_dir'].cleanup()
