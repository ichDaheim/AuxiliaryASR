import os
import re
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from text_normalizer import phonemize_with_lexicon

VALID_PHONEMES = {
    "a", "aː", "aɪ̯", "aʊ̯", "b", "d", "eː", "ɛ", "ɛː", "f", "g", "h",
    "i", "iː", "j", "k", "l", "m", "n", "ŋ", "oː", "ɔ", "øː", "p", "pf",
    "r", "s", "ʃ", "t", "ts", "uː", "ʊ", "v", "x", "ç", "yː", "y", "z", "ɐ", "ə"
}

ADDITIONAL_PHONEMES = {
    "<pad>", "<bos>","<eos>","<unk>"," ",",",".",":","!","?"
}

# Pfade zu den Dateien
input_file_path = '/mnt/d/Projekte/StyleTTS/AuxiliaryASR/Data/val_list.txt'
output_file_path = '/mnt/d/Projekte/StyleTTS/AuxiliaryASR/Data/val_list_processed.txt'
phones_lookup_path = '/mnt/d/Projekte/StyleTTS/AuxiliaryASR/Data/phones_lookup.txt'


def process_file_and_generate_phonemes(input_path, output_path, phones_path):
    """
    Liest eine Datei zeilenweise ein, fügt Suffixe hinzu, extrahiert Text,
    wandelt ihn in Phoneme um und speichert beides in separaten Dateien.
    """
    # Initialisiere den Phonemizer für Deutsch (einmalig für bessere Performance)
    try:
        phonemizer = EspeakBackend(
            language='de',
            preserve_punctuation=False,
            language_switch='remove-flags',   # verhindert Sprachwechsel
            punctuation_marks=';:,.!?¡¿—…"«»“”()[]{}',  # optional
            with_stress=False                 # wenn du keine Betonungszeichen willst
        )
    except Exception as e:
        print(f"Fehler beim Initialisieren von espeak-ng. Ist es installiert? Fehler: {e}")
        print("Stellen Sie sicher, dass 'espeak-ng' auf Ihrem System installiert ist (z.B. via 'sudo apt-get install espeak-ng' oder 'choco install espeak-ng').")
        return

    # Initialisiere das Set direkt mit den speziellen Tokens. Das ist effizienter.
    unique_phonemes = set(ADDITIONAL_PHONEMES)

    def phonemize_and_collect(text, line_context):
        """Hilfsfunktion zur Phonemisierung und Sammlung von Phonemen."""
        # Wende die Normalisierung an, bevor der Text an den Phonemizer geht
        normalized_text = phonemize_with_lexicon(text,phonemizer)
        phonemes_str = phonemizer.phonemize([normalized_text], strip=True, separator=Separator(phone=' ', word='  ', syllable=''))[0]

        # DEBUG: Gib eine Warnung aus, wenn '??' gefunden wird
        if '??' in phonemes_str:
            print(f"\n[WARNUNG] '??' in folgender Zeile gefunden:")
            print(f"  -> Zeile: {line_context}")
            print(f"  -> Text: '{text}'")
            print(f"  -> Phonemisiert: '{phonemes_str}'\n")

        all_tokens = phonemes_str.split()
        valid_phonemes = [p for p in all_tokens if not (p.startswith('(') and p.endswith(')')) and p != '1']
        unique_phonemes.update(valid_phonemes)

    try:
        # Öffne die Eingabedatei zum Lesen und die Ausgabedatei zum Schreiben
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            print("Verarbeite Zeilen und extrahiere Phoneme...")
            processed_lines = 0
            for line in infile:
                # Entferne führende/nachfolgende Leerzeichen und Zeilenumbrüche
                stripped_line = line.strip()

                # Überspringe leere Zeilen
                if not stripped_line:
                    continue

                # --- Aufgabe 1: Suffixe hinzufügen und in _processed.txt schreiben ---
                # --- Aufgabe 2: Phoneme nur aus den verwendeten Zeilen extrahieren ---
                if "_Ash_" in stripped_line:
                    outfile.write(f"{stripped_line}|0\n")
                    processed_lines += 1
                    
                    # Extrahiere Phoneme nur, wenn die Zeile verarbeitet wird
                    parts = stripped_line.split('|')
                    if len(parts) > 1:
                        phonemize_and_collect(parts[1], stripped_line)

                elif "_Alloy_" in stripped_line:
                    outfile.write(f"{stripped_line}|1\n")
                    processed_lines += 1

                    # Extrahiere Phoneme nur, wenn die Zeile verarbeitet wird
                    parts = stripped_line.split('|')
                    if len(parts) > 1:
                        phonemize_and_collect(parts[1], stripped_line)
                else:
                    print(f"line skipped: {stripped_line}")

        print(f"Verarbeitung der Hauptdatei abgeschlossen!")
        print(f"Insgesamt {processed_lines} Zeilen modifiziert.")
        print(f"Die neue Datei wurde hier gespeichert: {os.path.abspath(output_path)}")

        # --- Aufgabe 3: Einzigartige Phoneme in phones_lookup.txt schreiben ---
        if unique_phonemes:
            # Sortiere die Phoneme für eine konsistente Reihenfolge
            sorted_phonemes = sorted(list(unique_phonemes))
            phone_id = 0 # Startnummer für die Phonem-ID, wie im Beispiel
            with open(phones_path, 'w', encoding='utf-8') as phone_file:
                for phone in sorted_phonemes:
                    phone_file.write(f'"{phone}",{phone_id}\n')
                    phone_id += 1
            print(f"\nPhonem-Extraktion abgeschlossen!")
            print(f"{len(sorted_phonemes)} einzigartige Phoneme gefunden und in '{os.path.abspath(phones_path)}' gespeichert.")
        else:
            print("\nKeine Phoneme zum Speichern gefunden.")


    except FileNotFoundError:
        print(f"Fehler: Die Datei '{os.path.abspath(input_path)}' wurde nicht gefunden.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

# Führe die Funktion aus
if __name__ == "__main__":
    process_file_and_generate_phonemes(input_file_path, output_file_path, phones_lookup_path)