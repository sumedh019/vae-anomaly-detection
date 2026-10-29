"""Session-isolated temporary checkpoint uploads for hosted inspection."""
import hashlib
import tempfile
from pathlib import Path
from src.anomaly_detector import _load_model


def save_uploaded_checkpoint(data, session):
    if not data or len(data) > 200 * 1024 * 1024:
        raise ValueError('Checkpoint must be nonempty and no larger than 200 MB.')
    digest = hashlib.sha256(data).hexdigest()
    if session.get('_checkpoint_digest') == digest:
        path = Path(session['_checkpoint_dir'].name) / 'model.pth'
        if path.is_file():
            return path
    directory = tempfile.TemporaryDirectory(prefix='vae-session-')
    path = Path(directory.name) / 'model.pth'
    try:
        path.write_bytes(data)
        _load_model(path)
    except Exception:
        directory.cleanup()
        raise
    previous = session.get('_checkpoint_dir')
    session['_checkpoint_dir'] = directory
    session['_checkpoint_digest'] = digest
    if previous is not None:
        previous.cleanup()
    return path
