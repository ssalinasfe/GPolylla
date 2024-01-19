#!/bin/bash
#SBATCH --job-name=gpolylla           # Nombre del trabajo
#SBATCH --output=gpolylla_%j.out      # Nombre del archivo de salida
#SBATCH --error=gpolylla_%j.err       # Nombre del archivo de error
#SBATCH --ntasks=1                   # Numero de tareas
#SBATCH --partition=rtx
#SBATCH --gpus=1

date

EXECUTABLE_FOLDER=$1
OUTPUT_FOLDER=$2
NSAMPLES=$(($3-1))

START=1000000
END=100000000

# Definir el tamaño del problema

# Calculamos la diferencia entre los valores extremos y la cantidad de valores que queremos obtener
diff=$(($END-$START))
interval=$((diff/NSAMPLES))

echo "Start: $START, End: $END, Intervalo: $interval, Diferencia: $diff, NSAMPLES: $NSAMPLES"

# Inicializamos un array para almacenar los 5 valores equidistantes con raíz cuadrada entera
values=()

# Iteramos sobre los 5 valores equidistantes
for ((i=$START; i<=$END; i+=interval)); do
    # Calculamos la raíz cuadrada y su parte entera
    sqrt=$(bc <<< "scale=0;sqrt($i)")
    sqrt_int=$(bc <<< "$sqrt/1")

    # Inicializamos las variables para guardar el valor más cercano a la raíz cuadrada entera
    closest_lower=0
    closest_upper=0

    # Iteramos sobre los dos valores enteros más cercanos a la raíz cuadrada
    for ((j=sqrt_int-1; j<=sqrt_int+1; j++)); do
        # Calculamos el cuadrado del valor entero y su distancia al valor original
        j_squared=$((j*j))
        distance=$((j_squared-i))

        # Si este valor es más cercano que los anteriores, lo guardamos
        if [[ $distance -ge 0 ]] && ( [[ $distance -lt $((closest_upper-i)) ]] || [[ $closest_upper -eq 0 ]] ); then
            closest_upper=$j_squared
        elif [[ $distance -lt 0 ]] && ( [[ $((-distance)) -lt $((-closest_lower+i)) ]] || [[ $closest_lower -eq 0 ]] ); then
            closest_lower=$j_squared
        fi
    done

    # Si hay un valor más cercano en la dirección positiva, lo usamos
    if [[ $closest_upper -ne 0 ]] && [[ $((closest_upper-i)) -lt $((i-closest_lower)) ]]; then
        values+=($closest_upper)
    # Si hay un valor más cercano en la dirección negativa, lo usamos
    elif [[ $closest_lower -ne 0 ]]; then
        values+=($closest_lower)
    # Si no hay un valor más cercano en ninguna dirección, usamos el valor original
    else
        values+=($i)
    fi
done

echo "Valores: ${values[@]}"
echo "" >> gpolylla_${SLURM_JOB_ID}.out
echo "" >> gpolylla_${SLURM_JOB_ID}.err

# Llamamos al programa con los NSAMPLEs valores como argumentos
for value in "${values[@]}"; do
    date

    echo "Ejecutando Polylla con tamaño $value"

    # Medir el tiempo de ejecución del programa Polylla
    TIMEFORMAT="%E"
    ELAPSED_TIME=$(time (srun --container-workdir=${PWD} --container-name=cuda-12.1.0-brew -p rtx --gpus=1 ${EXECUTABLE_FOLDER}/Polylla -i $value ${OUTPUT_FOLDER}/grid_${value} >> ${OUTPUT_FOLDER}/gpolylla_${SLURM_JOB_ID}.out) 2>&1 >/dev/null)

    # Imprimir el tiempo de ejecución en los archivos de salida y error
    echo "Tiempo de ejecución: $ELAPSED_TIME segundos" >> ${OUTPUT_FOLDER}/Polylla_${SLURM_JOB_ID}.out
    echo "Tiempo de ejecución: $ELAPSED_TIME segundos" >> ${OUTPUT_FOLDER}/Polylla_${SLURM_JOB_ID}.err
    echo "" >> ${OUTPUT_FOLDER}/Polylla_${SLURM_JOB_ID}.out
    echo "" >> ${OUTPUT_FOLDER}/Polylla_${SLURM_JOB_ID}.err
done

date
