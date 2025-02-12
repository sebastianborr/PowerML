# script: check_libraries.py

import subprocess

# Ejecutar el comando pip list para obtener las librerías instaladas
installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "list"])

# Decodificar y mostrar la salida
print(installed_packages.decode("utf-8"))
