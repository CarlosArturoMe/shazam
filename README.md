# Audio Identifier System (SIA)

The file \_\_init\_\_.py, SIA allows to load from
one, or several folders songs in mp3, wav
or raw format to extract fingerprints from them using the Shazam fingerprint model, and store them in MySQL, PostgreSQL
or ElasticSearch.
Later, the recognizer\_test.py file is used
to choose a sample contained in a folder
and make a recognition of which song is
played through the computer's microphone.
Recognition tests can be performed
with different noise rates modifying
the constants ADD\_NOISE to True, and SNR 
to the desired value.
The confusion matrix, and response times
for each one are stored in CSV files.
You can also use recognizer.py to simply
recognize a played song that was previously stored.

Requirements: Python 3.7 and conda to install the libraries contained in the environment.yml file.
Install the database chosen and create the database music\_recognition to store the fingerprint tracks.
Scheme is automatically created executing
\_\_init\_\_.py.

# Sistema identificador de audio (SIA)

Mediante el archivo \_\_init\_\_.py, SIA permite cargar de 
una, o varias carpetas canciones en formato mp3, wav 
o raw para extraer huellas de ellas mediante el modelo 
de huella de Shazam, y almacenarlas en MySQL, PostgreSQL
o ElasticSearch.
Posteriormente se emplea el archivo recognizer\_test.py 
para elegir una muestra contenida en alguna carpeta
y realizar un reconocimiento de que canción se 
reproduce por medio del micrófono del equipo.
Se pueden realizar pruebas de reconocimiento
con disintas tasas de ruido modificando
las constantes ADD\_NOISE a True y SNR al valor deseado.
La matriz de confusión, y tiempos de respuesta
para cada una se almacenan en archivos CSV.
También se puede usar recognizer.py para simplemente
reconocer una canción reproducida que fue almacenada previamente.

Requerimientos: Python 3.7 y conda para instalar las librerias contenidas en el archivo environment.yml. 
Instalar la base de datos elegida y crear la BD music\_recognition para albergar las huellas.
El esquema se crea automáticamente al correr el
archivo \_\_init\_\_.py.