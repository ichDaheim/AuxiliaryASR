import re
import os
from phonemizer.separator import Separator

def load_lexicon(path):
    """Lädt das benutzerdefinierte Lexikon aus einer Datei."""
    lexicon = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word = parts[0].strip()
                    phonemes = parts[1].strip()
                    lexicon[word] = phonemes
    except FileNotFoundError:
        print(f"[WARNUNG] Lexikon-Datei nicht gefunden unter: {path}")
    return lexicon

# Lade das Lexikon einmal beim Start des Moduls
lexicon_path = os.path.join(os.path.dirname(__file__), 'custom_lexicon.txt')
CUSTOM_LEXICON = load_lexicon(lexicon_path)

def normalize_text(text):
    """
    Bereinigt den Text und bereitet ihn für die Phonemisierung vor.
    Verwendet ein benutzerdefiniertes Lexikon für bekannte Problemwörter.
    """
    text = text.lower()
    # Entferne alle Anführungszeichen und ähnliche Zeichen
    text = re.sub(r'["“”`´\']', '', text)
    # Normalisiere Gedankenstriche
    text = re.sub(r'[—–]', '-', text)
    return text

def phonemize_with_lexicon(text, phonemizer):
    """
    Phonemisiert einen Text, wobei ein benutzerdefiniertes Lexikon für Problemwörter verwendet wird.
    """
    normalized_text = normalize_text(text)
    words = normalized_text.split()

    phoneme_parts = []
    for word in words:
        # Entferne mögliche restliche Satzzeichen für den Lexikon-Lookup
        clean_word = re.sub(r'[,.!?]$', '', word)

        if clean_word in CUSTOM_LEXICON:
            phoneme_parts.append(CUSTOM_LEXICON[clean_word])
        else:
            # Phonemisiere nur Wörter, die nicht im Lexikon sind
            phonemes = phonemizer.phonemize([word], strip=True, separator=Separator(phone=' ', word='  '))[0]
            phoneme_parts.append(phonemes)

    return " ".join(phoneme_parts)