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
from recognizer import fingerprint

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

#FIELD_FILE_SHA1 = 'file_sha1'
FIELD_FILE_SHA1 = 2 #result is tuple, this field index is 2

#self.config = config
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

#@staticmethod
def get_file_fingerprints(file_name: str, limit: int, print_output: bool = False):
    channels, fs, file_hash = read(file_name, limit)
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
    try:
        file_name, limit = arguments
    except ValueError:
        pass
    
    song_name, extension = os.path.splitext(os.path.basename(file_name))

    fingerprints, file_hash = get_file_fingerprints(file_name, limit, print_output=False)

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
    #print("nprocesses: ",nprocesses)
    pool = multiprocessing.Pool(nprocesses)

    filenames_to_fingerprint = []
    for filename, _ in find_files(path, extensions):
        # don't refingerprint already fingerprinted files
        if unique_hash(filename) in songhashes_set:
            print(f"{filename} already fingerprinted, continuing...")
            continue

        filenames_to_fingerprint.append(filename)

    # Prepare _fingerprint_worker input
    worker_input = list(zip(filenames_to_fingerprint, [limit] * len(filenames_to_fingerprint)))
    #print("worker_input: ",worker_input)
    # Send off our tasks; returns iterable that yield the result of function passed
    #iterator = pool.imap_unordered(_fingerprint_worker, worker_input)

    """
    # Loop till we have all of them
    while True:
        try:
            song_name, hashes, file_hash = next(iterator)
            print("song_name after iterator: ",song_name)
            print("hashes: ",hashes)
            print("file_hash: ",file_hash)
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
            sid = db.insert_song(song_name, file_hash, len(hashes))
            print("sid",sid)
            db.insert_hashes(sid, hashes)
            db.set_song_fingerprinted(sid)
            #__load_fingerprinted_audio_hashes()
            songs = db.get_songs()
            songhashes_set = set()  # to know which ones we've computed before
            for song in songs:
                song_hash = song[FIELD_FILE_SHA1]
            songhashes_set.add(song_hash)
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

def load_fingerprinted_audio_hashes(songhashes_set):
    # to know which ones we've computed before
    songs = db.get_songs()
    #print("songs: ",songs)
    print("len(songs): ",len(songs))
    for song in songs:
        song_hash = song[FIELD_FILE_SHA1]
        songhashes_set.add(song_hash)
    return songhashes_set


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
fingerprint_directory("songs", ["." + "mp3"], 4,songhashes_set)