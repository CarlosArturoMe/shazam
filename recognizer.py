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

RECORD_SECONDS = 1
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

# Number of results being returned for file recognition
TOPN = 2

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

def generate_hashes(peaks, fan_value: int = DEFAULT_FAN_VALUE):
    """
    Hash list structure:
       sha1_hash[0:FINGERPRINT_REDUCTION]    time_offset
        [(e05b341a9b77a51fd26, 32), ... ]
    :param peaks: list of peak frequencies and times.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :return: a list of hashes with their corresponding offsets.
    """
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

def get_2D_peaks(arr2D: np.array, plot: bool = False, amp_min: int = DEFAULT_AMP_MIN):
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

def generate_fingerprints(samples, Fs=RATE):
    t = time()
    hashes = fingerprint(samples, Fs=Fs)
    #print("hashes: ",hashes)
    fingerprint_time = time() - t
    
    return hashes, fingerprint_time

def return_matches(hashes, batch_size: int = 1000):
    """
    Searches the database for pairs of (hash, offset) values.

    :param hashes: A sequence of tuples in the format (hash, offset)
        - hash: Part of a sha1 hash, in hexadecimal format
        - offset: Offset this hash was created from/at.
    :param batch_size: number of query's batches.
    :return: a list of (sid, offset_difference) tuples and a
    dictionary with the amount of hashes matched (not considering
    duplicated hashes) in each song.
        - song id: Song identifier
        - offset_difference: (database_offset - sampled_offset)
    """
    # Create a dictionary of hash => offset pairs for later lookups
    mapper = {}
    for hsh, offset in hashes:
        if hsh.upper() in mapper.keys():
            mapper[hsh.upper()].append(offset)
        else:
            mapper[hsh.upper()] = [offset]

    values = list(mapper.keys())
    #print("values = list(mapper.keys())",values)
    #print("len(values)",len(values))
    # in order to count each hash only once per db offset we use the dic below
    dedup_hashes = {}

    results = []
    with db.cursor() as cur:
        for index in range(0, len(values), batch_size):
            # Create our IN part of the query
            query = SELECT_MULTIPLE % ', '.join([IN_MATCH] * len(values[index: index + batch_size]))
            #print("query: ",query)
            #print(values[index: index + batch_size])
            cur.execute(query, values[index: index + batch_size])

            for hsh, sid, offset in cur:
                #print("hsh: ",hsh)
                if sid not in dedup_hashes.keys():
                    dedup_hashes[sid] = 1
                else:
                    dedup_hashes[sid] += 1
                #  we now evaluate all offset for each  hash matched
                #print("mapper: ",mapper)
                #print("hsh: ",hsh)
                for song_sampled_offset in mapper[hsh]:
                    results.append((sid, offset - song_sampled_offset))

        return results, dedup_hashes

def find_matches(hashes):
    """
    Finds the corresponding matches on the fingerprinted audios for the given hashes.

    :param hashes: list of tuples for hashes and their corresponding offsets
    :return: a tuple containing the matches found against the db, a dictionary which counts the different
        hashes matched for each song (with the song id as key), and the time that the query took.

    """
    t = time()
    matches, dedup_hashes = return_matches(hashes)
    query_time = time() - t

    return matches, dedup_hashes, query_time


def align_matches(matches, dedup_hashes, queried_hashes,topn: int = TOPN):
        """
        Finds hash matches that align in time with other matches and finds
        consensus about which hashes are "true" signal from the audio.

        :param matches: matches from the database
        :param dedup_hashes: dictionary containing the hashes matched without duplicates for each song
        (key is the song id).
        :param queried_hashes: amount of hashes sent for matching against the db
        :param topn: number of results being returned back.
        :return: a list of dictionaries (based on topn) with match information.
        """
        #DEBUGEAR ESTA FUNCION
        # count offset occurrences per song and keep only the maximum ones.
        sorted_matches = sorted(matches, key=lambda m: (m[0], m[1]))
        #print("sorted_matches: ",sorted_matches)
        counts = [(*key, len(list(group))) for key, group in groupby(sorted_matches, key=lambda m: (m[0], m[1]))]
        #print(counts)
        songs_matches = sorted(
            [max(list(group), key=lambda g: g[2]) for key, group in groupby(counts, key=lambda count: count[0])],
            key=lambda count: count[2], reverse=True
        ) #reverse=True = From higest to lowest
        #print("songs_matches: ",songs_matches)
        songs_result = []
        for song_id, offset, _ in songs_matches[0:topn]:  # consider topn elements in the result
            song = db.get_song_by_id(song_id)

            song_name = song.get(SONG_NAME, None)
            song_hashes = song.get(FIELD_TOTAL_HASHES, None)
            nseconds = round(float(offset) / DEFAULT_FS * DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO, 5)
            hashes_matched = dedup_hashes[song_id]

            song = {
                SONG_ID: song_id,
                SONG_NAME: song_name.encode("utf8"),
                INPUT_HASHES: queried_hashes,
                FINGERPRINTED_HASHES: song_hashes,
                HASHES_MATCHED: hashes_matched,
                # Percentage regarding hashes matched vs hashes from the input.
                INPUT_CONFIDENCE: round(hashes_matched / queried_hashes, 2),
                # Percentage regarding hashes matched vs hashes fingerprinted in the db.
                FINGERPRINTED_CONFIDENCE: round(hashes_matched / song_hashes, 2),
                OFFSET: offset,
                OFFSET_SECS: nseconds,
                FIELD_FILE_SHA1: song.get(FIELD_FILE_SHA1, None).encode("utf8")
            }

            songs_result.append(song)

        return songs_result

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

#MAIN
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True
                ,frames_per_buffer=CHUNK)
print ("recording...")
data_channels = [[] for i in range(CHANNELS)]
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK,exception_on_overflow = False)
    nums = np.fromstring(data, np.int16)
    frames.append(data)
    for c in range(CHANNELS):
        data_channels[c].extend(nums[c::CHANNELS])
print("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
#print(data_channels)

fingerprint_times = []
hashes = set()  # to remove possible duplicated fingerprints we built a set.
for channel in data_channels:
    fingerprints, fingerprint_time = generate_fingerprints(channel, Fs=RATE)
    fingerprint_times.append(fingerprint_time)
    hashes |= set(fingerprints) #union
#print("fingerprint_times: ",fingerprint_times)
#print("hashes: ",hashes)
db_cls = get_database(config.get("database_type", "mysql").lower())
db = db_cls(**config.get("database", {}))
matches, dedup_hashes, query_time = find_matches(hashes)
t = time()
final_results = align_matches(matches, dedup_hashes, len(hashes))
align_time = time() - t

print("final_results: ",final_results)
print("fingerprint_times: ",np.sum(fingerprint_times))
print("query_time: ",query_time)
print("align_time: ",align_time)
#return final_results, np.sum(fingerprint_times), query_time, align_time
"""
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

y, sr = librosa.load(WAVE_OUTPUT_FILENAME, sr=None, mono=True)
stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
#D = librosa.amplitude_to_db(stft, ref=np.max)
#librosa.display.specshow(D, sr=sr, hop_length=512, x_axis='time', y_axis='linear')
#plt.colorbar(format='%+2.0f dB')
#plt.show()
mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
mean_val = np.mean(f, axis=1)
max_val = np.max(f, axis=1)
min_val = np.min(f, axis=1)
print("mean: ",mean_val)
print("max: ",max_val)
print("min: ",min_val)

file_name = "119025"
dict_data = []
mean_n1 = mean_val[0]
mean_n2 = mean_val[1]
mean_n3 = mean_val[2]
mean_n4 = mean_val[3]
mean_n5 = mean_val[4]
mean_n6 = mean_val[5]
mean_n7 = mean_val[6]
mean_n8 = mean_val[7]
mean_n9 = mean_val[8]
mean_n10 = mean_val[9]
mean_n11 = mean_val[10]
mean_n12 = mean_val[11]
mean_n13 = mean_val[12]
mean_n14 = mean_val[13]
mean_n15 = mean_val[14]
mean_n16 = mean_val[15]
mean_n17 = mean_val[16]
mean_n18 = mean_val[17]
mean_n19 = mean_val[18]
mean_n20 = mean_val[19]
max_n1 = max_val[0]
max_n2 = max_val[1]
max_n3 = max_val[2]
max_n4 = max_val[3]
max_n5 = max_val[4]
max_n6 = max_val[5]
max_n7 = max_val[6]
max_n8 = max_val[7]
max_n9 = max_val[8]
max_n10 = max_val[9]
max_n11 = max_val[10]
max_n12 = max_val[11]
max_n13 = max_val[12]
max_n14 = max_val[13]
max_n15 = max_val[14]
max_n16 = max_val[15]
max_n17 = max_val[16]
max_n18 = max_val[17]
max_n19 = max_val[18]
max_n20 = max_val[19]
min_n1 = min_val[0]
min_n2 = min_val[1]
min_n3 = min_val[2]
min_n4 = min_val[3]
min_n5 = min_val[4]
min_n6 = min_val[5]
min_n7 = min_val[6]
min_n8 = min_val[7]
min_n9 = min_val[8]
min_n10 = min_val[9]
min_n11 = min_val[10]
min_n12 = min_val[11]
min_n13 = min_val[12]
min_n14 = min_val[13]
min_n15 = min_val[14]
min_n16 = min_val[15]
min_n17 = min_val[16]
min_n18 = min_val[17]
min_n19 = min_val[18]
min_n20 = min_val[19]
dict_data.append({"file_name":file_name,"max_n1":max_n1,"max_n2":max_n2,
"max_n3":max_n3,"max_n4":max_n4,"max_n5":max_n5,"max_n6":max_n6,
"max_n7":max_n7,"max_n8":max_n8,"max_n9":max_n9,"max_n10":max_n10,
"max_n11":max_n11,"max_n12":max_n12,"max_n13":max_n13,"max_n14":max_n14,
"max_n15":max_n15,"max_n16":max_n16,"max_n17":max_n17,"max_n18":max_n18,
"max_n19":max_n19,"max_n20":max_n20,"mean_n1":mean_n1,"mean_n2":mean_n2,
"mean_n3":mean_n3,"mean_n4":mean_n4,"mean_n5":mean_n5,"mean_n6":mean_n6,
"mean_n7":mean_n7,"mean_n8":mean_n8,"mean_n9":mean_n9,"mean_n10":mean_n10,
"mean_n11":mean_n11,"mean_n12":mean_n12,"mean_n13":mean_n13,"mean_n14":mean_n14,
"mean_n15":mean_n15,"mean_n16":mean_n16,"mean_n17":mean_n17,"mean_n18":mean_n18,
"mean_n19":mean_n19,"mean_n20":mean_n20,"min_n1":min_n1,"min_n2":min_n2,
"min_n3":min_n3,"min_n4":min_n4,"min_n5":min_n5,"min_n6":min_n6,
"min_n7":min_n7,"min_n8":min_n8,"min_n9":min_n9,"min_n10":min_n10,
"min_n11":min_n11,"min_n12":min_n12,"min_n13":min_n13,"min_n14":min_n14,
"min_n15":min_n15,"min_n16":min_n16,"min_n17":min_n17,"min_n18":min_n18,
"min_n19":min_n19,"min_n20":min_n20
})
csv_columns = ['file_name','max_n1','max_n2','max_n3','max_n4','max_n5','max_n6','max_n7','max_n8','max_n9','max_n10','max_n11'
,'max_n12','max_n13','max_n14','max_n15','max_n16','max_n17','max_n18',
'max_n19','max_n20','mean_n1','mean_n2','mean_n3','mean_n4','mean_n5','mean_n6'
,'mean_n7','mean_n8','mean_n9','mean_n10','mean_n11','mean_n12','mean_n13','mean_n14'
,'mean_n15','mean_n16','mean_n17','mean_n18','mean_n19','mean_n20',
'min_n1','min_n2','min_n3','min_n4','min_n5','min_n6','min_n7','min_n8','min_n9','min_n10','min_n11'
,'min_n12','min_n13','min_n14','min_n15','min_n16','min_n17','min_n18','min_n19','min_n20']
csv_file = file_name+".csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
except IOError:
    print("I/O error")
"""