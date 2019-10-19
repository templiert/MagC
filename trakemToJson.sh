#!/bin/bash

ABSOLUTE_SCRIPT=`readlink -m $0`
SCRIPTS_DIR=`dirname ${ABSOLUTE_SCRIPT}`
. ${SCRIPTS_DIR}/setup_java_env.sh 240G

trakemProjectPath=$1
dataPath=$2
outputJson=$3

runJavaCommandAndExit org.janelia.alignment.trakem2.Converter $trakemProjectPath $dataPath $outputJson $*
