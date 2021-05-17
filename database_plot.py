import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import csv
import importlib
import pandas as pd

RECORD_SECONDS = 5
# Number of results being returned for file recognition
TOPN = 5
CHANNELS = 2
RATE = 44100 #Hz, samples / second
CHUNK = 8192
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
# TABLE SONGS
SONGS_TABLENAME = "songs"
# SONGS FIELDS
FIELD_SONG_ID = 'song_id'
FIELD_SONGNAME = 'song_name'
FIELD_FINGERPRINTED = "fingerprinted"
FIELD_FILE_SHA1 = 'file_sha1'
FIELD_TOTAL_HASHES = 'total_hashes'
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

DATABASES = {
    'mysql': ("mysql_database", "MySQLDatabase"),
    'postgres': ("dejavu.database_handler.postgres_database", "PostgreSQLDatabase"),
    'elastic': ("elastic_database","ElasticDatabase")
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

#config = {'host': 'localhost', 'port': 9200}

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

db_cls = get_database(config.get("database_type", "mysql").lower())
db = db_cls(**config.get("database", {}))

#TOP 10 songs with more hashes
SELECT_HASHES_ASC_SONGS = f"""
        SELECT
           `{FIELD_SONGNAME}`
        ,   `{FIELD_TOTAL_HASHES}`
        FROM `{SONGS_TABLENAME}`
        ORDER BY `{FIELD_TOTAL_HASHES}` ASC
        LIMIT 10;
    """

SELECT_HASHES_DESC_SONGS = f"""
        SELECT
           `{FIELD_SONGNAME}`
        ,   `{FIELD_TOTAL_HASHES}`
        FROM `{SONGS_TABLENAME}`
        ORDER BY `{FIELD_TOTAL_HASHES}` DESC
        LIMIT 10;
    """

SELECT_SONGS_HASHES = f"""
        SELECT
           `{FIELD_SONGNAME}`
        ,   `{FIELD_TOTAL_HASHES}`
        FROM `{SONGS_TABLENAME}`
        ORDER BY `{FIELD_TOTAL_HASHES}` ASC
    """

with db.cursor(dictionary=True) as cur:
    cur.execute(SELECT_SONGS_HASHES)
    response = list(cur)
#print(response)

#songs_arr = []
#total_hashes_arr = []
#for song in response:
    #songs_arr.append(song[0]+".mp3")
    #total_hashes_arr.append(song[1])

"""    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(songs_arr, total_hashes_arr) #align='center')
#ax.xaxis.set_tick_params(labelsize='small')
#ax.text(fontsize=6)
plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=6)
ax.set_xlabel('Canciones')
ax.set_ylabel('Total de hashes')
plt.show()
"""
#save to csv
with open('song_hashes.csv', 'w',) as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Nombre', 'Cantidad de hashes'])
    for song in response:
        writer.writerow([song[0]+".mp3", song[1]])


