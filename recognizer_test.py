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
                # t_delta son ms
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
    #print(specgram[0])
    #for arr in specgram[0]:
    #    for num in arr:
    #        print(abs(num))
    #print("to take specgram")
    #REVISANDO AQUI
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
    # return hashes 
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

    # in order to count each hash only once per db offset we use the dic below
    dedup_hashes = {}

    results = []
    with db.cursor() as cur:

        for index in range(0, len(values), batch_size):
            # Create our IN part of the query
            query = SELECT_MULTIPLE % ', '.join([IN_MATCH] * len(values[index: index + batch_size]))

            cur.execute(query, values[index: index + batch_size])
            #matches_count= 0
            for hsh, sid, offset in cur:
                #matches_count +=1
                if sid not in dedup_hashes.keys():
                    dedup_hashes[sid] = 1
                else:
                    dedup_hashes[sid] += 1
                #  we now evaluate all offset for each  hash matched
                #print("mapper: ",mapper)
                #print("hsh: ",hsh)
                for song_sampled_offset in mapper[hsh]:
                    results.append((sid, offset - song_sampled_offset))
            #print("matches_count: ",matches_count)
        #print("Dedup hashes: ",dedup_hashes)
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
        songs_matches = sorted(
            [max(list(group), key=lambda g: g[2]) for key, group in groupby(counts, key=lambda count: count[0])],
            key=lambda count: count[2], reverse=True
        )
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
                #SONG_NAME: song_name.encode("utf8"),
                SONG_NAME: str(song_name),
                #how many hashes were created for the sample
                INPUT_HASHES: queried_hashes,
                #total hashes of song in DB - fingerprinted_hashes_in_db
                FINGERPRINTED_HASHES: song_hashes,
                #how many hashes are temporal aligned
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

def play_thread():
    work = True
    while work:
        fragment_from_song = q.get()
        if fragment_from_song is None:
            break
        play(fragment_from_song)
        work = False

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


#SNR in dB
#given a signal and desired SNR, this gives the required AWGN what should be added to the signal to get the desired SNR
def get_white_noise(signal,SNR):
    #RMS value of signal
    RMS_s=math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise=np.random.normal(0, STD_n, signal.shape[0])
    return noise

#given a signal, noise (audio) and desired SNR, this gives the noise (scaled version of noise input) that gives the desired SNR
def get_noise_from_sound(signal,noise,SNR):
    RMS_s=math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    
    #current RMS of noise
    RMS_n_current=math.sqrt(np.mean(noise**2))
    noise=noise*(RMS_n/RMS_n_current)
    
    return noise

def generate_csv_results(songs_to_recognize,recognized_song_names,iteration,final_results_arr):
    #print(songs_to_recognize)
    #print(recognized_song_names)
    #print(iteration)
    dict_data = []
    songs_to_recognize_just_name = []
    for i in range(len(songs_to_recognize)):
        #trackandfile = songs_to_recognize[i].replace('songs/', '')
        track_name = re.sub(r'^.*?/', '', songs_to_recognize[i])
        track_name = re.sub(r'^.*?/', '', track_name)
        song_to_recognize = track_name.replace('.mp3', '')
        songs_to_recognize_just_name.append(song_to_recognize)
        #print("song_to_recognize: ",song_to_recognize)
        #if re.search(recognized_song_names[i], str(songs_to_recognize[i])):
        if song_to_recognize == recognized_song_names[i]:
            dict_data.append({"file_name_played":str(songs_to_recognize[i]),"file_name_result":str(recognized_song_names[i]),
            "song_start_time":times[i]["song_start_time"],"correct":1,"fingerprint_times":times[i]["fingerprint_times"],
            "query_time":times[i]["query_time"],"align_time":times[i]["align_time"],"total_time":times[i]["total_time"]})
            #,"final_results":final_results_arr[i]}) #final_results_arr[i] must be string to save here
        else:
            dict_data.append({"file_name_played":str(songs_to_recognize[i]),"file_name_result":str(recognized_song_names[i]),"song_start_time":times[i]["song_start_time"],
            "correct":0,"fingerprint_times":times[i]["fingerprint_times"],"query_time":times[i]["query_time"],
            "align_time":times[i]["align_time"],"total_time":times[i]["total_time"]})
            #,"final_results":final_results_arr[i]})
        """
        #write csv with top results detail of file recognition
        if len(final_results_arr[i]) > 0:
            keys = final_results_arr[i][0].keys()
            with open(song_to_recognize+'_results.csv', 'w', newline='')  as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(final_results_arr[i])
        """

    csv_columns = ['file_name_played','file_name_result','song_start_time','correct','fingerprint_times','query_time','align_time',
    'total_time']
    #,'final_results']
    if add_noise:
        csv_name = "shazam_results_" + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "_" + str(len(songs_to_recognize)) + "records_" + str(RECORD_SECONDS) + "seconds_SNR" + SNR + "_atSong" + str(iteration+1) + ".csv"
    else:
        csv_name = "shazam_results_" + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "_" + str(len(songs_to_recognize)) + "records_" + str(RECORD_SECONDS) + "seconds" + "_atSong" + str(iteration+1) + ".csv"
    csv_file = csv_name
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
    #generate confusion matrix
    y_true = pd.Series(songs_to_recognize_just_name, name="Actual")
    y_pred = pd.Series(recognized_song_names)
    #df_confusion = pd.crosstab(y_true, y_pred)
    df_confusion = pd.crosstab(y_true, y_true)
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            df_confusion.at[y_true[i],y_true[i]] = str(0)
            df_confusion.at[y_true[i],y_pred[i]] = str(1)
    df_confusion.to_csv('CM_'+csv_name)
    #using scykit
    cm = confusion_matrix(songs_to_recognize_just_name, recognized_song_names)
    cr = classification_report(songs_to_recognize_just_name, recognized_song_names,output_dict=True)
    asc = accuracy_score(songs_to_recognize_just_name, recognized_song_names)
    print(cm)
    print(cr)
    print("Accuracy score: ",asc)
    df = pd.DataFrame(cm)
    df.to_csv('CMSK_'+csv_name)
    df2 = pd.DataFrame(cr).transpose()
    df2.to_csv('CRSK_'+csv_name)
    df3 = pd.DataFrame([asc])
    df3.to_csv('ASSK_'+csv_name)


#MAIN
songs_to_recognize = find_files("songsES",["." + "mp3"])
#songs_to_recognize = ["songsES/000/000002.mp3"]
songs_to_recognize = songs_to_recognize[0:100]
#song = songs_to_recognize[0]
recognized_song_names = []
times = []
final_results_arr = []
add_noise = False
SNR = 0
db_cls = get_database(config.get("database_type", "mysql").lower())
db = db_cls(**config.get("database", {}))

len_songs = len(songs_to_recognize)
fourthpart = math.floor(len_songs/4)
medium = fourthpart * 2
three_fourths = fourthpart * 3
print("len_songs: ",len_songs)
#print("fourthpart: ",fourthpart)
#print("medium: ",medium)
#print("three_fourths: ",three_fourths)
for song_i, song_name in enumerate(songs_to_recognize):
    print("Now loading: ",song_name)
    if add_noise:
        signal, sr = librosa.load(song_name)
        #additive gaussian noise
        #noise=get_white_noise(signal,SNR=10)
        signal=np.interp(signal, (signal.min(), signal.max()), (-1, 1))
        noise_file='city-traffic-sounds/city-traffic-sounds.mp3'
        noise, sr = librosa.load(noise_file)
        #print(len(noise))
        noise=np.interp(noise, (noise.min(), noise.max()), (-1, 1))
        if(len(noise)>len(signal)):
            noise=noise[0:len(signal)]
        noise=get_noise_from_sound(signal,noise,SNR)
        signal=signal+noise
        sf.write("signal_with_noise.wav", signal, sr)
        song_to_play = AudioSegment.from_wav("signal_with_noise.wav")
    else:
        song_to_play = AudioSegment.from_mp3(song_name)
    r_seconds = RECORD_SECONDS * 1000
    duration_seconds = song_to_play.duration_seconds
    #random start from 0 to duration of song - RECORD_SECONDS
    song_start_time = randrange(0,int(duration_seconds)-RECORD_SECONDS)
    print("start_time: ",song_start_time)
    start_time = song_start_time * 1000
    fragment_from_song = song_to_play[start_time:start_time+r_seconds]
    print("Song {} of {}".format(song_i,len_songs))
    #start playing song
    t = threading.Thread(target=play_thread)
    t.start()
    q.put(fragment_from_song)
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
    #print(data_channels)
    fingerprint_times = []
    hashes = set()  # to remove possible duplicated fingerprints we built a set.
    for channel in data_channels:
        fingerprints, fingerprint_time = generate_fingerprints(channel, Fs=RATE)
        fingerprint_times.append(fingerprint_time)
        hashes |= set(fingerprints) #union
    #print("fingerprint_times: ",fingerprint_times)
    #print("hashes: ",hashes)
    matches, dedup_hashes, query_time = find_matches(hashes)
    t = time()
    final_results = align_matches(matches, dedup_hashes, len(hashes))
    align_time = time() - t

    print("final_results: ",final_results)
    print("fingerprint_time: ",np.sum(fingerprint_times)) #total time for fingerprinting all the segments
    print("query_time: ",query_time)
    print("align_time: ",align_time)
    if final_results:
        final_results_arr.append(final_results)
        recognized_song_names.append(str(final_results[0]['song_name']))
    else:
        #final_results_arr.append("No results")
        final_results_arr.append([])
        recognized_song_names.append("No results")
    fingerprint_times = np.sum(fingerprint_times)
    total_time = fingerprint_times + query_time + align_time
    times.append({"song_start_time":song_start_time,"fingerprint_times":fingerprint_times,"query_time":query_time,
    "align_time":align_time,"total_time":total_time})
    #if song_i == fourthpart or song_i == medium or three_fourths == song_i or len_songs-1 == song_i:
    if len_songs-1 == song_i:
        generate_csv_results(songs_to_recognize[:song_i+1],recognized_song_names,song_i,final_results_arr)
audio.terminate()
