import importlib
import multiprocessing
import os
import sys
import traceback
import fnmatch
from hashlib import sha1
from pydub import AudioSegment
from pydub.utils import audioop
import numpy as np
#from recognizer import fingerprint
import matplotlib.mlab as mlab
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)
from scipy.ndimage.filters import maximum_filter
from operator import itemgetter
import hashlib
import matplotlib.pyplot as plt

# DATABASE CLASS INSTANCES:
DATABASES = {
    'mysql': ("mysql_database", "MySQLDatabase"),
    'postgres': ("dejavu.database_handler.postgres_database", "PostgreSQLDatabase")
}

config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",
        "password": "1234",
        "database": "music_recognition"
    },
    "database_type": "mysql"
}

#FIELD_FILE_SHA1 = 'file_sha1'
FIELD_FILE_SHA1 = 2 #result is tuple, this field index is 2
RATE = 44100 #Hz, samples / second
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

# initialize db
def get_database(database_type: str = "mysql"):
    """
    Given a database type it returns a database instance for that type.

    :param database_type: type of the database.
    :return: an instance of BaseDatabase depending on given database_type.
    """
    try:
        path, db_class_name = DATABASES[database_type]
        db_module = importlib.import_module(path)
        db_class = getattr(db_module, db_class_name)
        return db_class
    except (ImportError, KeyError):
        raise TypeError("Unsupported database type supplied.")

#fingerprint_directory file methods
def read(file_name: str, limit: int = None):
    """
    Reads any file supported by pydub (ffmpeg) and returns the data contained
    within. If file reading fails due to input being a 24-bit wav file,
    wavio is used as a backup.

    Can be optionally limited to a certain amount of seconds from the start
    of the file by specifying the `limit` parameter. This is the amount of
    seconds from the start of the file.

    :param file_name: file to be read.
    :param limit: number of seconds to limit.
    :return: tuple list of (channels, sample_rate, content_file_hash).
    """
    # pydub does not support 24-bit wav files, use wavio when this occurs
    try:
        audiofile = AudioSegment.from_file(file_name)

        if limit:
            audiofile = audiofile[:limit * 1000]
        #warning here
        data = np.fromstring(audiofile.raw_data, np.int16)

        channels = []
        for chn in range(audiofile.channels):
            channels.append(data[chn::audiofile.channels])

        audiofile.frame_rate
    except audioop.error:
        print("audioop.error")
    """
        _, _, audiofile = wavio.readwav(file_name)

        if limit:
            audiofile = audiofile[:limit * 1000]

        audiofile = audiofile.T
        audiofile = audiofile.astype(np.int16)

        channels = []
        for chn in audiofile:
            channels.append(chn)
    """
    return channels, audiofile.frame_rate, unique_hash(file_name)


def get_2D_peaks(arr2D: np.array, plot: bool = False, amp_min: int = DEFAULT_AMP_MIN):
    #print("enter get 2D peaks")
    # Original code from the repo is using a morphology mask that does not consider diagonal elements
    # as neighbors (basically a diamond figure) and then applies a dilation over it, so what I'm proposing
    # is to change from the current diamond figure to a just a normal square one:
    #       F   T   F           T   T   T
    #       T   T   T   ==>     T   T   T
    #       F   T   F           T   T   T
    # In my local tests time performance of the square mask was ~3 times faster
    # respect to the diamond one, without hurting accuracy of the predictions.
    # I've made now the mask shape configurable in order to allow both ways of find maximum peaks.
    # That being said, we generate the mask by using the following function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    #all elements are neighbors
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)

    #  And then we apply dilation (dilating with itself) using the following function
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
    #  Take into account that if PEAK_NEIGHBORHOOD_SIZE is 2 you can avoid the use of the scipy functions and just
    #  change it by the following code:
    #  neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter mask (maximum filter uses erosion with neighborhood as STREL Struct Elem)
    #maximum filter hace más delgada una imagen
    # erosion removes thin lines, isolated dots. Dilation fattens up
    #print(maximum_filter(arr2D, footprint=neighborhood) == arr2D)
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D #Returns true where numpy array is equal to the erotioned https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
    #print("local_max: ",local_max)
    # Applying erosion, the dejavu documentation does not talk about this step. ?
    #print((arr2D == 0))
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    #print(eroded_background)
    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten() #Return a copy of the array collapsed into one dimension.
    #print(amps)
    # get indices for frequency and time
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(times_filter, freqs_filter)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return list(zip(freqs_filter, times_filter))

def generate_hashes(peaks, fan_value: int = DEFAULT_FAN_VALUE):
    """
    Hash list structure:
       sha1_hash[0:FINGERPRINT_REDUCTION]    time_offset
        [(e05b341a9b77a51fd26, 32), ... ]
    :param peaks: list of peak frequencies and times.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :return: a list of hashes with their corresponding offsets.
    """
    #print("enter generate hashes")
    # frequencies are in the first position of the tuples
    idx_freq = 0
    # times are in the second position of the tuples
    idx_time = 1

    if PEAK_SORT:
        peaks.sort(key=itemgetter(1))

    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                freq1 = peaks[i][idx_freq]
                freq2 = peaks[i + j][idx_freq]
                t1 = peaks[i][idx_time]
                t2 = peaks[i + j][idx_time]
                t_delta = t2 - t1
                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    h = hashlib.sha1(f"{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode('utf-8'))
                    hashes.append((h.hexdigest()[0:FINGERPRINT_REDUCTION], t1))
    
    return hashes

def fingerprint(channel_samples,
                Fs: int = RATE,
                wsize: int = DEFAULT_WINDOW_SIZE,
                wratio: float = DEFAULT_OVERLAP_RATIO,
                fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN):
    """
    FFT the channel, log transform output, find local maxima, then return locally sensitive hashes.

    :param channel_samples: channel samples to fingerprint.
    :param Fs: audio sampling rate.
    :param wsize: FFT windows size.
    :param wratio: ratio by which each sequential window overlaps the last and the next window.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    REVISAR
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list of hashes with their corresponding offsets.
    """
    #print("Enter fingerprint")
    # FFT the signal and extract frequency components, 0 is the spectrum
    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]
    #[0] Is the angle spectrum in radians.
    # Apply log transform since specgram function returns linear array. 0s are excluded to avoid np warning.
    #Convert a power spectrogram (amplitude squared) to decibel (dB) units
    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))
    #print("arr2D: ",arr2D)
    local_maxima = get_2D_peaks(arr2D, plot=False, amp_min=amp_min)
    #print("local_maxima: ",local_maxima) 
    return generate_hashes(local_maxima, fan_value=fan_value)


def get_file_fingerprints(file_name: str, limit: int, print_output: bool = False):
    channels, fs, file_hash = read(file_name, limit)
    #print("Enter get_file_fingerprints")
    #print("channels:",channels)
    #print("fs: ",fs)
    #print("file_hash: ",file_hash)
    fingerprints = set()
    channel_amount = len(channels)
    for channeln, channel in enumerate(channels, start=1):
        if print_output:
            print(f"Fingerprinting channel {channeln}/{channel_amount} for {file_name}")

        hashes = fingerprint(channel, Fs=fs)
        #print("hashes: ",hashes)
        if print_output:
            print(f"Finished channel {channeln}/{channel_amount} for {file_name}")

        fingerprints |= set(hashes)
    #print("fingerprints: ",fingerprints)
    #print("file_hash: ",file_hash)
    return fingerprints, file_hash

#@staticmethod
def _fingerprint_worker(arguments):
    # Pool.imap sends arguments as tuples so we have to unpack
    # them ourself.
    #print("Enter _fingerprint_worker")
    try:
        file_name, limit = arguments
    except ValueError:
        print("Error in _fingerprint_worker")
        pass
    
    song_name, extension = os.path.splitext(os.path.basename(file_name))

    fingerprints, file_hash = get_file_fingerprints(file_name, limit, print_output=True)

    return song_name, fingerprints, file_hash

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
                results.append((p, extension))
    return results

def unique_hash(file_path: str, block_size: int = 2**20):
    """ Small function to generate a hash to uniquely generate
    a file. Inspired by MD5 version here:
    http://stackoverflow.com/a/1131255/712997

    Works with large files.

    :param file_path: path to file.
    :param block_size: read block size.
    :return: a hash in an hexagesimal string form.
    """
    s = sha1()
    with open(file_path, "rb") as f:
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            s.update(buf)
    return s.hexdigest().upper()

def fingerprint_directory(path: str, extensions: str, nprocesses: int = None, songhashes_set=set()):
    """
    Given a directory and a set of extensions it fingerprints all files that match each extension specified.

    :param path: path to the directory.
    :param extensions: list of file extensions to consider.
    :param nprocesses: amount of processes to fingerprint the files within the directory.
    """
    # Try to use the maximum amount of processes if not given.
    try:
        nprocesses = nprocesses or multiprocessing.cpu_count()
    except NotImplementedError:
        nprocesses = 1
    else:
        nprocesses = 1 if nprocesses <= 0 else nprocesses
    print("nprocesses: ",nprocesses)
    pool = multiprocessing.Pool(nprocesses)

    filenames_to_fingerprint = []
    for filename, _ in find_files(path, extensions):
        # don't refingerprint already fingerprinted files
        if unique_hash(filename) in songhashes_set:
            print(f"{filename} already fingerprinted, continuing...")
            continue
        filenames_to_fingerprint.append(filename)

    # Prepare _fingerprint_worker input
    #print([limit] * len(filenames_to_fingerprint))
    worker_input = list(zip(filenames_to_fingerprint, [limit] * len(filenames_to_fingerprint)))
    #print("worker_input: ",worker_input)
    # Send off our tasks; returns iterable that yield the result of function passed
    iterator = pool.imap_unordered(_fingerprint_worker, worker_input)
    
    # Loop till we have all of them
    while True:
        try:
            song_name, hashes, file_hash = next(iterator)
            print("song_name after iterator: ",song_name)
            #print("hashes: ",hashes)
            #print("file_hash: ",file_hash)
        except multiprocessing.TimeoutError:
            print("multiprocessing.TimeoutError: ")
            continue
        except StopIteration:
            print("StopIteration")
            break
        except Exception:
            print("Failed fingerprinting")
            # Print traceback because we can't reraise it here
            traceback.print_exc(file=sys.stdout)
        else:
            print("finish hashes of song: ",song_name)
            sid = db.insert_song(song_name, file_hash, len(hashes))
            db.insert_hashes(sid, hashes)
            db.set_song_fingerprinted(sid)
            songhashes_set = load_fingerprinted_audio_hashes(songhashes_set)
    pool.close()
    pool.join() 
    """
    for obj in worker_input:
        song_name, hashes, file_hash = _fingerprint_worker(obj)
        print("song_name after _fingerprint_worker: ",song_name)
        #print("hashes: ",hashes)
        #print("file_hash: ",file_hash)
        sid = db.insert_song(song_name, file_hash, len(hashes))
        #print("sid",sid)
        db.insert_hashes(sid, hashes)
        db.set_song_fingerprinted(sid)
    songhashes_set = load_fingerprinted_audio_hashes(songhashes_set)
    """

def load_fingerprinted_audio_hashes(songhashes_set):
    # to know which ones we've computed before
    songs = db.get_songs()
    #print("songs: ",songs)
    print("len(songs): ",len(songs))
    for song in songs:
        song_hash = song[FIELD_FILE_SHA1]
        songhashes_set.add(song_hash)
    return songhashes_set

if __name__ == '__main__':
    db_cls = get_database(config.get("database_type", "mysql").lower())
    db = db_cls(**config.get("database", {}))
    #db is mysql instance
    with db.cursor() as cur:
        cur.execute(db.CREATE_SONGS_TABLE)
        cur.execute(db.CREATE_FINGERPRINTS_TABLE)
        cur.execute(db.DELETE_UNFINGERPRINTED)

    # if we should limit seconds fingerprinted,
    # None|-1 means use entire track
    limit = config.get("fingerprint_limit", None)
    if limit == -1:  # for JSON compatibility
        limit = None
    songhashes_set = load_fingerprinted_audio_hashes(set())
    fingerprint_directory("songsES", ["." + "mp3"], 4,songhashes_set)
