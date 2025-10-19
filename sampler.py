import os
import numpy as np
import soundfile as sf
from torch.utils.data import Sampler
from tqdm import tqdm

class BucketBatchSampler(Sampler):
    """
    Sorts the dataset by length and creates batches of similar-length sequences.
    This minimizes padding and significantly reduces memory usage.

    It also caches the audio lengths to a file to speed up subsequent runs.
    """
    def __init__(self, dataset, batch_size, drop_last=False, cache_path="lengths.npy"):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cache_path = cache_path

        if os.path.exists(self.cache_path):
            print(f"BucketBatchSampler: Lade Audio-Längen aus Cache-Datei: {self.cache_path}")
            lengths = np.load(self.cache_path)
            if len(lengths) != len(dataset):
                print("BucketBatchSampler: Cache ist veraltet. Berechne Längen neu.")
                lengths = self._calculate_lengths()
        else:
            lengths = self._calculate_lengths()

        # Sortiere die Indizes des Datasets basierend auf den Längen
        self.sorted_indices = np.argsort(lengths)

    def _calculate_lengths(self):
        """
        Calculates the length of each audio file in the dataset.
        This is a one-time operation.
        """
        print("BucketBatchSampler: Berechne die Längen aller Audiodateien (dies ist eine einmalige Operation)...")
        
        lengths = np.zeros(len(self.dataset), dtype=int)
        # Verwende eine einzige geöffnete Datei für Effizienz
        with open(self.dataset.data_path, 'r', encoding='utf-8') as f:
            for i in tqdm(range(len(self.dataset))):
                f.seek(self.dataset.line_offsets[i])
                line = f.readline().strip()
                path = line.split('|')[0]
                try:
                    # soundfile.info ist viel schneller als sf.read, da es nur den Header liest
                    info = sf.info(path)
                    lengths[i] = info.frames
                except Exception as e:
                    print(f"Warnung: Konnte Länge für Datei {path} nicht lesen: {e}. Setze Länge auf 0.")
                    lengths[i] = 0

        np.save(self.cache_path, lengths)
        print(f"BucketBatchSampler: Längenberechnung abgeschlossen und in {self.cache_path} gespeichert.")
        return lengths

    def __iter__(self):
        # Erstelle Batches aus den sortierten Indizes
        batches = [self.sorted_indices[i:i + self.batch_size] for i in range(0, len(self.sorted_indices), self.batch_size)]

        if self.drop_last and len(self.sorted_indices) % self.batch_size > 0:
            batches = batches[:-1]

        # Mische die Reihenfolge der Batches, aber nicht den Inhalt der Batches selbst
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
