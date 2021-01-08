import pyaudio
import wave
import librosa
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import librosa.display
import csv
from time import time
import matplotlib.mlab as mlab
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)
from scipy.ndimage.filters import maximum_filter
import hashlib
from operator import itemgetter
from itertools import groupby
import importlib
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import threading
import os
import fnmatch
from time import sleep
import queue
q = queue.Queue()
from random import randrange
import math
import datetime
import re
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

RECORD_SECONDS = 5
# Number of results being returned for file recognition
TOPN = 5
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100 #Hz, samples / second
CHUNK = 8192
audio = pyaudio.PyAudio()
WAVE_OUTPUT_FILENAME = "file.wav"
DEFAULT_WINDOW_SIZE = 4096  #The number of data points used in each block for the FFT
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 5
DEFAULT_AMP_MIN = 10 #DB
CONNECTIVITY_MASK = 2
PEAK_NEIGHBORHOOD_SIZE = 10
PEAK_SORT = True
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 20
DEFAULT_FS = 44100

FIELD_HASH = 'hash'
FIELD_SONG_ID = 'song_id'
FIELD_OFFSET = 'offset'
FINGERPRINTS_TABLENAME = "fingerprints"
SONG_ID = "song_id"
SONG_NAME = 'song_name'
FIELD_TOTAL_HASHES = 'total_hashes'
# Hashes generated from the input.
INPUT_HASHES = 'input_total_hashes'
# Percentage regarding hashes matched vs hashes from the input.
INPUT_CONFIDENCE = 'input_confidence'
# Hashes fingerprinted in the db.
FINGERPRINTED_HASHES = 'fingerprinted_hashes_in_db'
HASHES_MATCHED = 'hashes_matched_in_input'
# Percentage regarding hashes matched vs hashes fingerprinted in the db.
FINGERPRINTED_CONFIDENCE = 'fingerprinted_confidence'
OFFSET = 'offset'
OFFSET_SECS = 'offset_seconds'
FIELD_FILE_SHA1 = 'file_sha1'

SELECT_MULTIPLE = f"""
        SELECT HEX(`{FIELD_HASH}`), `{FIELD_SONG_ID}`, `{FIELD_OFFSET}`
        FROM `{FINGERPRINTS_TABLENAME}`
        WHERE `{FIELD_HASH}` IN (%s);
    """
IN_MATCH = f"UNHEX(%s)"


# DATABASE CLASS INSTANCES:
DATABASES = {
    'mysql': ("mysql_database", "MySQLDatabase"),
    'postgres': ("dejavu.database_handler.postgres_database", "PostgreSQLDatabase")
}

config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",
        "password": "12345678",
        "database": "music_recognition"
    },
    "database_type": "mysql"
}

def find_files(path: str, extensions):
    """
    Get all files that meet the specified extensions.

    :param path: path to a directory with audio files.
    :param extensions: file extensions to look for.
    :return: a list of tuples with file name and its extension.
    """
    # Allow both with ".mp3" and without "mp3" to be used for extensions
    extensions = [e.replace(".", "") for e in extensions]

    results = []
    for dirpath, dirnames, files in os.walk(path):
        for extension in extensions:
            for f in fnmatch.filter(files, f"*.{extension}"):
                p = os.path.join(dirpath, f)
                results.append(p)
    return results

#MAIN
songs_to_recognize = find_files("songs",["." + "mp3"])
songs_to_delete = []
for song_i, song_name in enumerate(songs_to_recognize):
    print("Now loading: ",song_name)
    try:
        song_to_play = AudioSegment.from_mp3(song_name)
    except:
        print("Cant play: ",song_name)
        songs_to_delete.append(song_name)
    duration_seconds = song_to_play.duration_seconds
    r_seconds = RECORD_SECONDS * 1000
    #random start from 5s (0) to duration of song - RECORD_SECONDS
    if int(duration_seconds) - RECORD_SECONDS < 1:
        print("Not enough duration of song: ",duration_seconds) 
        songs_to_delete.append(song_name)
df = pd.DataFrame(songs_to_delete, columns=["songs_name"])
df.to_csv('songs_to_delete.csv', index=False)