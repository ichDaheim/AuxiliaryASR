# coding: utf-8

import soundfile as sf
import os
import os.path as osp
import time
import random
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from text_normalizer import phonemize_with_lexicon
from sampler import BucketBatchSampler

np.random.seed(1)
random.seed(1)
DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt')
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}


def load_duration_map(filepath='dataset.txt'):
    """
    Lädt die dataset.txt und erstellt ein Mapping von Dateinamen zu Dauer.
    Gibt None zurück, wenn die Datei nicht gefunden wird.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Datei zur Dauer-Filterung nicht gefunden: {filepath}. Filterung wird übersprungen.")
        return None

    duration_map = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)  # Überspringe die Kopfzeile
            for line in f:
                parts = line.strip().split(';')
                if len(parts) == 2:
                    duration_str, filename = parts
                    # Konvertiere deutsches Komma zu Punkt für die float-Umwandlung
                    duration = float(duration_str.replace(',', '.'))
                    duration_map[filename] = duration
    except Exception as e:
        logger.error(f"Fehler beim Lesen der Dauer-Datei {filepath}: {e}")
        return None
    return duration_map


class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 dict_path=DEFAULT_DICT_PATH,
                 sr=24000
                 ):
        
        # --- Lazy Loading Implementierung ---
        self.data_path = data_path
        self.line_offsets = []
        with open(data_path, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line.encode('utf-8'))
        # --- Ende Lazy Loading ---
        
        self.phonemizer = None # Nur deklarieren, nicht initialisieren
        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS
        
        self.dict_path = dict_path
        self.text_cleaner = None # Nur deklarieren, nicht initialisieren
        self.sr = sr
 
        self.to_melspec = None
        self.mean, self.std = -4, 4

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        # --- Lazy Initialisierung des Phonemizers ---
        if self.phonemizer is None:
            # LOKALER IMPORT: Nur im Worker-Prozess ausführen
            from phonemizer.backend import EspeakBackend
            self.phonemizer = EspeakBackend(
                language='de',
                preserve_punctuation=False,
                language_switch='remove-flags',
                punctuation_marks=';:,.!?¡¿—…"«»“”()[]{}',
                with_stress=False
            )

        if self.text_cleaner is None:
            # LOKALER IMPORT: Nur im Worker-Prozess ausführen
            from text_utils import TextCleaner
            self.text_cleaner = TextCleaner(self.dict_path)
        if self.to_melspec is None:
            # LOKALER IMPORT: Nur im Worker-Prozess ausführen
            import torchaudio
            self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
        # --- Ende ---
        # --- Zeile bei Bedarf lesen ---
        with open(self.data_path, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().strip()
        
        parts = line.split('|')
        data = parts if len(parts) == 3 else (*parts, 0)
        # --- Ende ---
        wave, text_tensor, speaker_id = self._load_tensor(data)
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_melspec(wave_tensor)

        if (text_tensor.size(0) + 1) >= (mel_tensor.size(1) // 3):
            mel_tensor = F.interpolate(
                mel_tensor.unsqueeze(0), size=(text_tensor.size(0) + 1) * 3, align_corners=False,
                mode='linear').squeeze(0)

        acoustic_feature = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std

        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        return wave_tensor, acoustic_feature, text_tensor, data[0]

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data

        # DEBUG: Gib den Pfad der aktuell verarbeiteten Datei aus
        #print(f"Lade Datei: {wave_path}")

        speaker_id = int(speaker_id)
        wave, sr = sf.read(wave_path)
        
        from phonemizer.separator import Separator

        normalized_text = phonemize_with_lexicon(text, self.phonemizer)
        # phonemize the text (Deutsch)
        phoneme_str = self.phonemizer.phonemize([normalized_text], strip=True,
                                                separator=Separator(phone=' ', word='  ', syllable=''))[0]

        # phonemizer gibt einen String zurück, z. B. "ˈhallo"
        # oder Silben mit Leerzeichen getrennt, z. B. "ˈh a l oː"

        # optional: vereinheitlichen / Leerzeichen entfernen
        phoneme_list = phoneme_str.split()

        # evtl. Apostrophe entfernen, falls welche vorkommen
        phoneme_list = [p for p in phoneme_list if p != "'"]

        # Text-Cleaner + Mapping in Indizes
        text = self.text_cleaner(phoneme_list)

        blank_index = self.text_cleaner.word_index_dictionary[" "]
        text.insert(0, blank_index)  # silence at beginning
        text.append(blank_index)

        text = torch.LongTensor(text)

        return wave, text, speaker_id


class Collater(object):
    """
    Args:
      return_wave (bool): if true, will return the wave data along with spectrogram.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ['' for _ in range(batch_size)]
        for bid, (_, mel, text, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            assert (text_size < (mel_size // 2))

        if self.return_wave:
            waves = [b[0] for b in batch]
            return texts, input_lengths, mels, output_lengths, paths, waves

        return texts, input_lengths, mels, output_lengths


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    # --- NEUE FILTERLOGIK ---
    # Anmerkung: Die Filterlogik ist mit dem Lazy-Loading-Ansatz nicht mehr direkt kompatibel,
    # da wir nicht mehr die ganze Liste im Speicher haben.
    # Es wäre besser, eine vorgefilterte Metadaten-Datei zu erstellen.
    # Ich lasse den Code vorerst auskommentiert, damit du siehst, was sich ändert.
    # duration_map = load_duration_map('./datacheck/dataset.txt')
    # if duration_map is not None:
    #     original_count = len(path_list)
    #
    #     # Filtere die path_list basierend auf der Dauer
    #     filtered_list = []
    #     for item in path_list:
    #         # Extrahiere den Dateinamen aus dem Pfad in der Liste
    #         filename = os.path.basename(item.split('|')[0])
    #         duration = duration_map.get(filename)
    #
    #         if duration is not None and duration <= 15.0:
    #             filtered_list.append(item)
    #
    #     logger.info(
    #         f"Dauer-Filterung: {original_count} Dateien gefunden, {len(filtered_list)} Dateien nach Filter (<= 15s) behalten.")
    #     path_list = filtered_list
    # --- ENDE DER FILTERLOGIK ---

    # Übergib nur den Pfad, nicht die ganze Liste
    dataset = MelDataset(path_list, **dataset_config) 

    # --- NEU: BucketBatchSampler verwenden ---
    # Anstatt `shuffle`, was inkompatibel ist, erstellen wir einen Sampler.
    batch_sampler = None
    if not validation:
        # cache_path ist der Dateiname, in dem die Längen gespeichert werden, um zukünftige Starts zu beschleunigen.
        batch_sampler = BucketBatchSampler(dataset, batch_size=batch_size, drop_last=True, cache_path="lengths.npy")

    collate_fn = Collater(**collate_config)
    # --- KORREKTUR: Logik für DataLoader-Argumente ---
    if batch_sampler is not None:
        # Wenn wir unseren BucketBatchSampler verwenden, dürfen wir batch_size, shuffle, drop_last nicht übergeben.
        data_loader = DataLoader(dataset,
                                 batch_sampler=batch_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'))
    else:
        # Standard-DataLoader für die Validierung (ohne Bucket-Sortierung)
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'))
    return data_loader
