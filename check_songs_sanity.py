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

RECORD_SECONDS = 15

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
songs_to_recognize = find_files("songs/027",["." + "mp3"])
songs_to_delete = []
for song_i, song_name in enumerate(songs_to_recognize):
    print("Now loading: ",song_name)
    duration_seconds = 0
    try:
        song_to_play = AudioSegment.from_mp3(song_name)
        duration_seconds = song_to_play.duration_seconds
    except:
        print("Cant play: ",song_name)
        songs_to_delete.append([song_name,0])
    if int(duration_seconds) > 1200:
        print("Duration of song: ",duration_seconds)
        print("song_name: ",song_name)
        songs_to_delete.append([song_name,duration_seconds])
        #os.remove(song_name)
        name = os.path.splitext(os.path.basename(song_name))[0]
        os.rename(song_name, "/home/labcic/Documentos/fma_full/"+name)
print("len(songs_to_delete): ",len(songs_to_delete))
if len(songs_to_delete) > 0:
    df = pd.DataFrame(np.array(songs_to_delete), columns=["song_name","duration"])
    df.to_csv('songs_deleted.csv', index=False)
