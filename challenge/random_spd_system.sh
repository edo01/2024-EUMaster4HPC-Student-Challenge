#!/bin/bash

##ml purge
##ml intel/2023a
##ml KAROLINA/FAKEintel
module load intel

PROGRAM="random_spd_system"
##FLAGS="-DMKL_ILP64 -qmkl-ilp64=parallel"
##LINKS="-qmkl-ilp64=parallel"

LINKS="-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
FLAGS="-I${MKLROOT}/include"

icpx -O2 ${FLAGS} src/${PROGRAM}.cpp -o ${PROGRAM} $LINKS

./${PROGRAM} "$@"

rm -f ${PROGRAM}
