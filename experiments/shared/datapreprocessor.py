"""
Data Preprocessing Module for EEG Pipeline with Metric Learning.

- EEG Data Loading
- Preprocessing (filtering, resampling)
- Noise Generation (gaussian, powerline, emg)
- Dataset Building with novelty detection support
"""

import os
import re
import logging
from loguru import logger
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# ==============================================================================
# STRUCTURED LOGGING
# ==============================================================================
def get_logger(name: str, log_file: str = None):
    """Get a structured logger with optional file output."""
    logger.remove()
    logger.add(lambda msg: print(msg, end=""),
               format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    if log_file:
        logger.add(log_file, rotation="10 MB", serialize=True)
    return logger.bind(module=name)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    """Configuration for EEG pipeline."""

    # Data
    data_path: str = "./dataset/"
    subfolder: str = "Segmented_Data"
    electrodes: List[str] = field(
        default_factory=lambda: ["T7", "F8", "Cz", "P4"]
    )
    sfreq: int = 200

    # Preprocessing
    filter_low: float = 1.0
    filter_high: float = 40.0

    # Dataset
    window_size: int = 800   # sfreq * 4
    step_size: int = 400     # sfreq * 2
    snr_levels: Tuple[int, ...] = (20, 10, 5, 0)

    # Novelty detection holdout subjects
    holdout_subjects: List[int] = field(
        default_factory=lambda: [2, 5, 7, 12]
    )

    # Training
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 7

    # Model
    embed_dim: int = 128
    dropout: float = 0.3

    # Denoiser
    wavenet_blocks: int = 3
    wavenet_layers_per_block: int = 4

    # Metric learning
    arcface_margin: float = 0.3
    arcface_scale: float = 30.0

    # Output
    log_file: str = "output.json"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def n_channels(self) -> int:
        return len(self.electrodes)


# ==============================================================================
# DATA LOADING
# ==============================================================================
class EEGDataLoader:
    """Load and parse EEG data from CSV files."""

    FILENAME_PATTERN = re.compile(
        r'^(?P<subject>s\d{2})_(?P<experiment>ex\d{2})(?P<session>_s\d{2})?.csv$'
    )

    def __init__(self, config: Config, logger_inst=None):
        self.config = config
        self.logger = logger_inst or get_logger("eeg.dataloader")

    def parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse subject, experiment, session from filename."""
        match = self.FILENAME_PATTERN.match(filename)
        if not match:
            return None
        info = match.groupdict()
        info['subject'] = int(info['subject'][1:])
        info['experiment'] = int(info['experiment'][2:])
        info['session'] = int(info['session'][2:]) if info['session'] else None
        return info

    def load(self) -> Tuple[Dict, Dict]:
        """Load EEG data from folder."""
        folder_path = os.path.join(self.config.data_path, self.config.subfolder)
        eeg_data, eeg_df = {}, {}
        file_count = 0

        self.logger.info("load_start", folder=folder_path)

        if not os.path.isdir(folder_path):
            alt_folder = os.path.join(self.config.data_path, "Filtered_Data")
            if os.path.isdir(alt_folder):
                self.logger.warning(
                    "subfolder_missing_using_fallback",
                    expected=folder_path, fallback=alt_folder,
                )
                folder_path = alt_folder
            else:
                raise FileNotFoundError(
                    f"Dataset folder not found: {folder_path}. "
                    f"Expected subfolder '{self.config.subfolder}' under "
                    f"data_path='{self.config.data_path}'. "
                    "Run `python download_dataset.py` (or set Config(data_path=...))."
                )

        for file in os.listdir(folder_path):
            if not file.endswith('.csv'):
                continue

            file_info = self.parse_filename(file)
            if not file_info:
                continue

            subject, experiment = file_info['subject'], file_info['experiment']
            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(file_path)[self.config.electrodes].T.to_numpy(copy=True)
                info = mne.create_info(
                    ch_names=self.config.electrodes,
                    sfreq=self.config.sfreq,
                    ch_types='eeg',
                )
                raw = mne.io.RawArray(df, info, verbose=False)

                if subject not in eeg_data:
                    eeg_data[subject] = {}
                    eeg_df[subject] = {}
                if experiment not in eeg_data[subject]:
                    eeg_data[subject][experiment] = []
                    eeg_df[subject][experiment] = []

                eeg_data[subject][experiment].append(raw)
                eeg_df[subject][experiment].append(df)
                file_count += 1

            except Exception as e:
                self.logger.error("file_load_failed", filename=file, error=str(e))

        self.logger.info("load_complete", subjects=len(eeg_df), files=file_count)
        return eeg_data, eeg_df


# ==============================================================================
# PREPROCESSING
# ==============================================================================
class EEGPreprocessor:
    """Preprocess EEG data with filtering and resampling."""

    def __init__(self, config: Config, logger_inst=None):
        self.config = config
        self.logger = logger_inst or get_logger("eeg.preprocessor")

    def preprocess(self, eeg_data: Dict) -> Dict:
        """Apply bandpass filter and resample."""
        preprocessed = {}
        total_processed = 0

        self.logger.info(
            "preprocess_start",
            filter_low=self.config.filter_low,
            filter_high=self.config.filter_high,
        )

        for subject in eeg_data:
            preprocessed[subject] = {}
            for experiment in eeg_data[subject]:
                preprocessed_exp = []
                for raw in eeg_data[subject][experiment]:
                    raw.filter(
                        self.config.filter_low, self.config.filter_high,
                        fir_design='firwin', verbose=False,
                    )
                    raw.resample(self.config.sfreq, verbose=False)
                    preprocessed_exp.append(raw)
                    total_processed += 1
                preprocessed[subject][experiment] = preprocessed_exp

        self.logger.info("preprocess_complete", total=total_processed)
        return preprocessed

    def to_numpy(self, processed_data: Dict) -> Dict:
        """Convert processed MNE data to numpy arrays."""
        result = {}
        for subject, experiments in processed_data.items():
            result[subject] = {}
            for experiment, raws in experiments.items():
                result[subject][experiment] = [raw.get_data() for raw in raws]
        return result


# ==============================================================================
# NOISE GENERATOR
# ==============================================================================
class NoiseGenerator:
    """Generate different types of noise with specified SNR."""

    NOISE_TYPES = ["gaussian", "powerline", "emg"]

    def __init__(self, config: Config):
        self.config = config

    def add_noise(self, clean_epoch: np.ndarray, snr_db: float,
                  noise_type: str = "gaussian") -> np.ndarray:
        """Add noise to clean epoch at specified SNR."""
        generators = {
            "gaussian": self._add_gaussian_noise,
            "powerline": self._add_powerline_noise,
            "emg": self._add_emg_noise,
        }
        if noise_type not in generators:
            raise ValueError(f"Unknown noise type: {noise_type}")
        return generators[noise_type](clean_epoch, snr_db)

    def _add_gaussian_noise(self, clean: np.ndarray, snr_db: float) -> np.ndarray:
        signal_power = np.mean(clean ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / (snr_linear + 1e-8)
        noise = np.random.randn(*clean.shape)
        noise = noise / (np.sqrt(np.mean(noise ** 2) + 1e-8)) * np.sqrt(noise_power + 1e-8)
        return clean + noise

    def _add_powerline_noise(self, clean: np.ndarray, snr_db: float,
                             freq: float = 50.0) -> np.ndarray:
        C, T = clean.shape
        t = np.arange(T) / self.config.sfreq
        phases = np.random.uniform(0, 2 * np.pi, size=(C, 1))
        sine = np.sin(2 * np.pi * freq * t + phases)
        signal_power = np.mean(clean ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / (snr_linear + 1e-8)
        sine = sine / (np.sqrt(np.mean(sine ** 2) + 1e-8)) * np.sqrt(noise_power + 1e-8)
        return clean + sine

    def _add_emg_noise(self, clean: np.ndarray, snr_db: float,
                       low: float = 20.0, high: float = 80.0) -> np.ndarray:
        C, T = clean.shape
        noise = np.random.randn(C, T)
        freqs = np.fft.rfftfreq(T, d=1.0 / self.config.sfreq)
        noise_fft = np.fft.rfft(noise, axis=-1)
        band_mask = (freqs >= low) & (freqs <= high)
        noise_fft *= band_mask[np.newaxis, :]
        noise_band = np.fft.irfft(noise_fft, n=T, axis=-1)
        signal_power = np.mean(clean ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / (snr_linear + 1e-8)
        noise_band = noise_band / (np.sqrt(np.mean(noise_band ** 2) + 1e-8)) * np.sqrt(noise_power + 1e-8)
        return clean + noise_band


# ==============================================================================
# DATASET BUILDER
# ==============================================================================
class EEGDatasetBuilder:
    """Build PyTorch datasets from processed EEG data."""

    def __init__(self, config: Config, logger_inst=None):
        self.config = config
        self.noise_generator = NoiseGenerator(config)
        self.label_encoder = LabelEncoder()
        self.logger = logger_inst or get_logger("eeg.dataset")

    def build_dataset_with_novelty(self, clean_df: Dict,
                                   noise_type: str = "gaussian"):
        """Build dataset with known/unknown subject split for novelty detection."""
        all_subjects = sorted(clean_df.keys())
        holdout_set = set(self.config.holdout_subjects)

        known_subjects = [s for s in all_subjects if s not in holdout_set]
        unknown_subjects = [s for s in all_subjects if s in holdout_set]

        self.logger.info(
            "novelty_split",
            train_subjects=list(known_subjects),
            holdout_subjects=list(unknown_subjects),
        )

        # Build known subjects dataset
        X_n_known, X_c_known, y_known = [], [], []
        for subject in known_subjects:
            if subject not in clean_df:
                continue
            X_n, X_c, y = self._process_subject(clean_df[subject], subject, noise_type)
            X_n_known.extend(X_n)
            X_c_known.extend(X_c)
            y_known.extend(y)

        # Build unknown subjects dataset
        X_n_unknown, X_c_unknown, y_unknown = [], [], []
        for subject in unknown_subjects:
            if subject not in clean_df:
                continue
            X_n, X_c, y = self._process_subject(clean_df[subject], subject, noise_type)
            X_n_unknown.extend(X_n)
            X_c_unknown.extend(X_c)
            y_unknown.extend(y)

        # Convert to tensors
        X_n_known = torch.tensor(np.array(X_n_known), dtype=torch.float32)
        X_c_known = torch.tensor(np.array(X_c_known), dtype=torch.float32)
        y_known_arr = np.array(y_known)
        y_known_enc = self.label_encoder.fit_transform(y_known_arr)
        y_known_t = torch.tensor(y_known_enc, dtype=torch.long)
        num_known = len(self.label_encoder.classes_)

        X_n_unknown = torch.tensor(np.array(X_n_unknown), dtype=torch.float32)
        X_c_unknown = torch.tensor(np.array(X_c_unknown), dtype=torch.float32)
        y_unknown_arr = np.array(y_unknown)

        self.logger.info(
            "novelty_dataset_complete",
            known_samples=len(y_known_t), unknown_samples=len(y_unknown_arr),
            num_known_classes=num_known,
        )

        return (X_n_known, X_c_known, y_known_t, num_known), \
               (X_n_unknown, X_c_unknown, y_unknown_arr)

    def _process_subject(self, sessions: Dict, subject: int,
                         noise_type: str) -> Tuple[list, list, list]:
        """Process a single subject's data into windowed noisy/clean pairs."""
        X_noisy, X_clean, y = [], [], []
        for exp in sorted(sessions.keys()):
            trials = sessions[exp]
            for trial in trials:
                n_samples = trial.shape[1]
                if n_samples < self.config.window_size:
                    continue
                for start in range(0, n_samples - self.config.window_size + 1,
                                   self.config.step_size):
                    clean_epoch = trial[:, start:start + self.config.window_size]
                    snr_db = np.random.choice(self.config.snr_levels)
                    noisy_epoch = self.noise_generator.add_noise(
                        clean_epoch, snr_db, noise_type
                    )
                    X_clean.append(clean_epoch)
                    X_noisy.append(noisy_epoch)
                    y.append(subject)
        return X_noisy, X_clean, y
