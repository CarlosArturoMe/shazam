import queue
from elasticsearch import Elasticsearch, ElasticsearchException,client,helpers
import binascii
import codecs
import uuid
import json
import base64

# SONGS INDEX
SONGS_INDEXNAME = "songs"
# SONGS FIELDS
FIELD_SONGNAME = 'song_name'
FIELD_FINGERPRINTED = "fingerprinted"
FIELD_FILE_SHA1 = 'file_sha1'
FIELD_TOTAL_HASHES = 'total_hashes'

# FINGERPRINTS INDEX
FINGERPRINTS_INDEXNAME = "fingerprints"
# FINGERPRINTS FIELDS
FIELD_SONG_ID = 'song_id'
FIELD_HASH = 'hash'
FIELD_OFFSET = 'offset'


class ElasticDatabase():
    type = "elastic"

    def __init__(self, **options):
        conn = Elasticsearch(**options)
        # Ping the connection before using it from the cache.
        if conn.ping():
            print('Connection succesful!!')
        else:
            print('Connection unsuccesful')
        self._options = options
        self.cursor = conn

    def create_songs_index(self):
        created = False
        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "max_result_window" : 25000
            },
            "mappings": {
                "properties": {
                    FIELD_SONGNAME: {
                        "type": "text"
                    },
                    FIELD_FINGERPRINTED: {
                        "type": "boolean"
                    },
                    FIELD_FILE_SHA1: {
                        "type": "binary"
                    },
                    FIELD_TOTAL_HASHES: {
                        "type": "integer"
                    }
                }
            }
        }
        try:
            if not self.cursor.indices.exists(SONGS_INDEXNAME):
                # Ignore 400 means to ignore "Index Already Exist" error.
                self.cursor.indices.create(index=SONGS_INDEXNAME, ignore=400, body=settings)
                print('Created Index')
                created = True
        except Exception as ex:
            print(str(ex))
        finally:
            return created

    def create_fingerprints_index(self):
        #ES doesn't seems to validate _size or binary
        created = False
        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    FIELD_HASH: {
                        "type": "binary",
                        "_size":20
                    },
                    FIELD_SONG_ID: {
                        "type": "_id"
                    },
                    FIELD_OFFSET: {
                        "type": "integer"
                    }
                }
                #,
                #"dynamic":"strict"
            }
        }
        try:
            if not self.cursor.indices.exists(FINGERPRINTS_INDEXNAME):
                # Ignore 400 means to ignore "Index Already Exist" error.
                self.cursor.indices.create(index=FINGERPRINTS_INDEXNAME, ignore=400, body=settings)
                print('Created Index')
            created = True
        except Exception as ex:
            print(str(ex))
        finally:
            return created

    def delete_unfingerprinted(self):
        body = {
            "query": {
                "match": {
                FIELD_FINGERPRINTED: False
                }
            }     
        }
        self.cursor.delete_by_query(SONGS_INDEXNAME,body)

    
    def setup(self):
        """
        Called on creation or shortly afterwards.
        """
        self.create_songs_index()
        self.create_fingerprints_index()
        self.delete_unfingerprinted()

    def set_song_fingerprinted(self, song_id):
        """
        Sets a specific song as having all fingerprints in the database.

        :param song_id: song identifier.
        """
        print("song_id to set fingerprinted: ",song_id)
        record = {
            "doc": {
            FIELD_FINGERPRINTED: True
            },
            "doc_as_upsert": True     
        }
        self.cursor.update(index=SONGS_INDEXNAME, id=song_id, body=record)

    def gen_dicts(self,values):
        #print("Values received: ",values)
        for val in values:
            #print("Val of the batch: ",val)
            #print("Typeof",type(val[1]))
            #binary_string = binascii.unhexlify(val[1])
            #b64 = codecs.encode(codecs.decode(val[1], 'hex'), 'base64').decode().replace("\n", "")
            #b64 = bytearray.fromhex(val[1]).decode()
            #print(binary_string)
            #data = val[1]
            #json_str = json.dumps( data )
            #json_bytes = json_str.encode('utf-8')
            #print ("json_bytes:", json_bytes)
            #print ("type json_bytes:", type(json_bytes), "\n")
            # TypeError: a bytes-like object is required, not 'dict'
            #encoded_data = base64.b64encode( json_bytes )
            #print ("encoded_data:", encoded_data)
            #print ("type encoded_data:", type(encoded_data), "\n")
            # cast the bytes object as a string
            #encoded_str = str( encoded_data )
            # remove b'' to avoid UnicodeDecodeError
            #encoded_str = encoded_str[2:-1]
            #print("encoded_str: ",encoded_str)
            yield {
                "_index": FINGERPRINTS_INDEXNAME,
                FIELD_SONG_ID: val[0],
                FIELD_HASH: val[1],
                FIELD_OFFSET: val[2]
            }

    def insert_hashes(self, song_id: int, hashes, batch_size: int = 1000):
        """
        Insert a multitude of fingerprints.

        :param song_id: Song identifier the fingerprints belong to
        :param hashes: A sequence of tuples in the format (hash, offset)
            - hash: Part of a sha1 hash, in hexadecimal format
            - offset: Offset this hash was created from/at.
        :param batch_size: insert batches.
        """
        values = [(song_id, hsh, int(offset)) for hsh, offset in hashes]
        for index in range(0, len(hashes), batch_size):
            helpers.bulk(self.cursor, self.gen_dicts(values[index: index + batch_size]))

    def find_matches(self, hashes):
        """
        Find coincident hashes
        :param hashes: A batch of hashes to find
            - hash: Part of a sha1 hash, in hexadecimal format
        SCAN RETURNED 274 hits
        """
        queries = []
        for hsh in hashes:
            #dec = codecs.encode(codecs.decode(hsh, 'hex'), 'base64').decode().replace("\n", "")
            #dec = str(binascii.unhexlify(hsh))
            #print("hsh: ",hsh)
            #term, match 
            queries.append({'match':{FIELD_HASH:hsh}})
        #res= self.cursor.search(index=FINGERPRINTS_INDEXNAME,body={'query':{"bool":{"should":queries}}
        res= helpers.scan(self.cursor,index=FINGERPRINTS_INDEXNAME,query={'query':{"bool":{"should":queries}},
        "fields": [FIELD_HASH, FIELD_SONG_ID, FIELD_OFFSET]})
        return res

    def insert_song(self, song_name: str, file_hash: str, total_hashes: int) -> int:
        """
        Inserts a song name into the database, returns the new
        identifier of the song.

        :param song_name: The name of the song.
        :param file_hash: Hash from the fingerprinted file.
        :param total_hashes: amount of hashes to be inserted on fingerprint table.
        :return: the inserted id.
        """
        try:
            record = {FIELD_SONGNAME:song_name,FIELD_FILE_SHA1:file_hash,FIELD_TOTAL_HASHES:total_hashes,FIELD_FINGERPRINTED:False}
            outcome = self.cursor.index(index=SONGS_INDEXNAME, body=record)
        except Exception as ex:
            print('Error indexing data')
            print(str(ex))
        return outcome['_id']


    def get_songs(self):
        """
        Returns all fully fingerprinted songs in the database

        :return: a dictionary with the songs info.
        """
        search_object = {'query': {'term': {FIELD_FINGERPRINTED: True}}, "fields": [FIELD_SONGNAME, FIELD_FILE_SHA1,
        FIELD_TOTAL_HASHES]}
        response = self.cursor.search(index = SONGS_INDEXNAME, body=search_object, size=25000)
        print("get_songs response: ",response)
        arr = []
        for hit in response["hits"]["hits"]:
            dct = {"song_name":hit['_source'][FIELD_SONGNAME],"total_hashes":hit['_source'][FIELD_TOTAL_HASHES],
                "file_sha1":hit['_source'][FIELD_FILE_SHA1]}
            arr.append(dct)
        return arr

    def get_song_by_id(self, song_id: int):
        """
        Brings the song info from the database.

        :param song_id: song identifier.
        :return: a song by its identifier. Result must be a Dictionary.
        """
        #print("song_id: ",song_id)
        search_object = {'query': {'term': {"_id": song_id}}, "fields": [FIELD_SONGNAME, FIELD_FILE_SHA1, FIELD_TOTAL_HASHES]}
        response = self.cursor.search(index=SONGS_INDEXNAME, body=search_object)
        #print("response: ",response)
        dct = {"song_name":response["hits"]["hits"][0]['_source'][FIELD_SONGNAME],
            "total_hashes":response["hits"]["hits"][0]['_source'][FIELD_TOTAL_HASHES],
            "file_sha1":response["hits"]["hits"][0]['_source'][FIELD_FILE_SHA1]}
        #print("dct: ",dct)
        return dct

    #Not used from here and below..

    def __getstate__(self):
        return self._options,

    def __setstate__(self, state):
        self._options, = state
        self.cursor = cursor_factory(**self._options)


def cursor_factory(**factory_options):
    def cursor(**options):
        options.update(factory_options)
        return Cursor(**options)
    return cursor


class Cursor(object):
    """
    Establishes a connection to the database and returns an open cursor.
    # Use as context manager
    with Cursor() as cur:
        cur.execute(query)
        ...
    """
    def __init__(self, dictionary=False, **options):
        super().__init__()
        self._cache = queue.Queue(maxsize=5)
        try:
            conn = self._cache.get_nowait()
            # Ping the connection before using it from the cache.
            #conn.ping(True)
            if conn.ping():
                print('Yay Connect')
            else:
                print('Awww it could not connect!')
        except queue.Empty:
            conn = Elasticsearch(**options)

        self.conn = conn
        self.dictionary = dictionary

    @classmethod
    def clear_cache(cls):
        cls._cache = queue.Queue(maxsize=5)

    def __enter__(self):
        self.cursor = self.conn
        #self.cursor = self.conn.cursor(dictionary=self.dictionary)
        return self.cursor

    def __exit__(self, extype, exvalue, traceback):
        print(extype)
        # if we had a ES related error we try to rollback the cursor.
        if extype is ElasticsearchException:
            self.cursor.rollback()

        self.cursor.close()
        #self.conn.commit()
        client.IndicesClient.flush(self.conn)
        # Put it back on the queue
        try:
            self._cache.put_nowait(self.conn)
        except queue.Full:
            self.conn.close()
    
