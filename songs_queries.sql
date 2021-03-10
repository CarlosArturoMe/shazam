SELECT * FROM music_recognition.songs;
SELECT COUNT(*) FROM music_recognition.songs;
SELECT * FROM music_recognition.songs where song_name like '%nandoo1%';

DELETE FROM music_recognition.songs where song_id = 8004;
DELETE FROM music_recognition.songs where song_id = 3523;

describe music_recognition.fingerprints;
SELECT song_name, total_hashes FROM songs ORDER BY total_hashes ASC LIMIT 25;
SELECT song_name, total_hashes FROM songs ORDER BY total_hashes DESC; 
SELECT AVG(total_hashes) from songs; #11,000 