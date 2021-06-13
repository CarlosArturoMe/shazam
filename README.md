# Sistema identificador de audio (SIA)

Medinate el archivo __init__.py SIA permite cargar de 
una, o varias carpetas canciones en formato mp3, wav 
o raw para extraer huellas de ellas mediante el modelo 
de huella de Shazam, y almacenarlas en MySQL, PostgreSQL
o ElasticSearch.
Posteriormente se emplea el archivo recognizer_test.py 
para elegir una muestra contenida en alguna carpeta
y realizar un reconocimiento de que canción se 
reproduce por medio del micrófono del equipo.
La matriz de confusión, y tiempos de respuesta
para cada una se almacenan en archivos CSV.

Requerimientos: Python 3.7 y conda para instalar las librerias contenidas en el archivo environment.yml. 
Instalar y crear la BD music_recognition en la base de datos elegida para albergar las huellas.