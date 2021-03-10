SELECT * FROM music_recognition.fingerprints LIMIT 100;
SELECT count(*) FROM music_recognition.fingerprints; # 95527890
SELECT * FROM music_recognition.fingerprints where song_id = 098619;
#SELECT * FROM music_recognition.fingerprints where song_id = 3523;
DELETE FROM music_recognition.fingerprints where song_id = 8002;
