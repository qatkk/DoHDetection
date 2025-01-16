import subprocess
import os

def run_bash_script_with_wsl():
    current_dir = os.path.dirname(__file__)
    bash_script = os.path.join(current_dir, 'bash.sh')

    try:
        result = subprocess.run(['C:/Program Files/Git/bin/bash.exe', bash_script], capture_output=True, text=True, check=True) #Change it depending on your OS
        print("Output del script:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error al ejecutar el script:", e.stderr)
        print("Código de error:", e.returncode)
        print("Salida estándar:", e.stdout)
        print("Salida de error:", e.stderr)
