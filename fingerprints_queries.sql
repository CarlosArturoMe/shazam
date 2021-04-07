SELECT * FROM music_recognition.fingerprints LIMIT 100;
SELECT count(*) FROM music_recognition.fingerprints; # 95527890 / 7986 = 11962
#con BD de 5568 canciones: 436682654
SELECT count(*) FROM music_recognition.fingerprints where song_id = 553; 
SELECT * FROM music_recognition.fingerprints where song_id = 386;
#SELECT * FROM music_recognition.fingerprints where song_id = 3523;
DELETE FROM music_recognition.fingerprints where song_id = 309;