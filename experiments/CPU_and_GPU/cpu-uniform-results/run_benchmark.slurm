#!/bin/bash
#SBATCH --job-name=gpolylla           # Nombre del trabajo
#SBATCH --output=gpolylla_%j.out      # Nombre del archivo de salida
#SBATCH --error=gpolylla_%j.err       # Nombre del archivo de error
#SBATCH --ntasks=1                   # Numero de tareas
#SBATCH --partition=rtx
#SBATCH --gpus=1

# run as:
# sbatch run_benchmark.slurm /home/rcarrasco/GPolylla/build /home/rcarrasco/GPolylla/data /home/rcarrasco/GPolylla/experiments/CPU_and_GPU/gpu-uniform-result/1 32

date

FOLDER=$1
INPUT=$2
OUTPUT=$3
NSAMPLES=$(($4))

START=1000000
END=44312500

# Definir el tamaño del problema

# Calculamos la diferencia entre los valores extremos y la cantidad de valores que queremos obtener
diff=$(($END-$START))
interval=$((diff/NSAMPLES))

echo "Start: $START, End: $END, Intervalo: $interval, Diferencia: $diff, NSAMPLES: $NSAMPLES"

# Llamamos al programa con los NSAMPLEs valores como argumentos
for ((i=$START; i<=$END; i+=interval)); do
    date

    echo "Ejecutando Polylla con tamaño $i"

    # Medir el tiempo de ejecución del programa Polylla
    TIMEFORMAT="%E"
    ELAPSED_TIME=$(time (srun --container-name=cuda-12.1.0-brew -p rtx --gpus=1 ${FOLDER}/Polylla -s ${INPUT}/${i}_points.1.off -o ${OUTPUT}/uniform_${i} >> ${OUTPUT}/Polylla_${SLURM_JOB_ID}.out) 2>&1 >/dev/null)

    # Imprimir el tiempo de ejecución en los archivos de salida y error
    echo "Tiempo de ejecución: $ELAPSED_TIME segundos" >> ${OUTPUT}/Polylla_${SLURM_JOB_ID}.out
    echo "Tiempo de ejecución: $ELAPSED_TIME segundos" >> ${OUTPUT}/Polylla_${SLURM_JOB_ID}.err
    echo "" >> ${OUTPUT}/Polylla_${SLURM_JOB_ID}.out
    echo "" >> ${OUTPUT}/Polylla_${SLURM_JOB_ID}.err
done

date