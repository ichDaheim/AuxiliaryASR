import os
import soundfile as sf
import glob
from tqdm import tqdm

# --- KONFIGURATION ---
# Geben Sie hier den Pfad zu Ihrem Ordner mit den .wav-Dateien an
AUDIO_FOLDER = 'D:/Projekte/HiFTNet_prep/checked/'

# Name der Ausgabedatei
OUTPUT_FILE = 'dataset.txt'


# --- ENDE DER KONFIGURATION ---

def analyze_audio_files(folder_path, output_filename):
    """
    Analysiert alle .wav-Dateien in einem Ordner, ermittelt ihre Länge
    und schreibt das Ergebnis in eine Textdatei.
    """
    # Finde alle .wav-Dateien im angegebenen Ordner und seinen Unterordnern
    # Der Parameter recursive=True sorgt dafür, dass auch Unterordner durchsucht werden.
    wav_files = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)

    if not wav_files:
        print(f"Keine .wav-Dateien im Ordner '{folder_path}' gefunden.")
        return

    print(f"{len(wav_files)} .wav-Dateien gefunden. Analysiere Längen...")

    try:
        # Öffne die Ausgabedatei zum Schreiben
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Schreibe die Kopfzeile
            f.write("Länge der Audiodatei in Sekunden;Dateiname\n")

            # Iteriere durch alle gefundenen Dateien mit einer Fortschrittsanzeige
            for file_path in tqdm(wav_files, desc="Verarbeite Audiodateien"):
                try:
                    # Lese die Metadaten der Audiodatei, ohne sie komplett zu laden
                    info = sf.info(file_path)
                    duration_seconds = info.duration

                    # Extrahiere nur den Dateinamen aus dem kompletten Pfad
                    file_name = os.path.basename(file_path)

                    # Formatiere die Dauer und ersetze den Punkt durch ein Komma
                    duration_str = f"{duration_seconds:.4f}".replace('.', ',')

                    # Schreibe die formatierte Zeile in die Datei
                    # Wir verwenden ein Semikolon als Trennzeichen, wie von Ihnen gewünscht.
                    f.write(f"{duration_str};{file_name}\n")

                except Exception as e:
                    print(f"\nFehler bei der Verarbeitung der Datei '{file_path}': {e}")

        print(f"\nAnalyse abgeschlossen. Ergebnisse wurden in '{os.path.abspath(output_filename)}' gespeichert.")

    except Exception as e:
        print(f"\nEin unerwarteter Fehler ist aufgetreten: {e}")


# Führe die Hauptfunktion aus
if __name__ == "__main__":
    analyze_audio_files(AUDIO_FOLDER, OUTPUT_FILE)