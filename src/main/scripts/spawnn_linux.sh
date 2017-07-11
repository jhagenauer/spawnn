#!/bin/sh

## set the path to your spawnn installation here or in an environment variable
#SPAWNN_HOME=${HOME}

## set the path to the files with additional operators here or in an environment variable
#SPAWNN_OPERATORS_ADDITIONAL=

if [ -z "${SPAWNN_HOME}" ] ; then
    BIN_DIR=`dirname "$0"`
    SPAWNN_HOME=`dirname "${BIN_DIR}"`
fi 

# JAVA_HOME set, so use it
if [ ! -z "${JAVA_HOME}" ] ; then
	if [ -x "${JAVA_HOME}/bin/java" ]; then
		JAVA="${JAVA_HOME}/bin/java"
	fi
fi
# otherwise, try to find java using which
if [ -z "${JAVA}" ] ; then
	_jfnd="`which java`"
	if [ -x "${_jfnd}" ]; then
		JAVA="${_jfnd}"
	else
		echo 'Could not find the java executable in default path or ${JAVA_HOME}/bin/java.'
		echo "Edit $0 and/or your local startup files."
		exit 1
	fi
	unset _jfnd
fi

for JAR in ${SPAWNN_HOME}/lib/*.jar 
do
CLASSPATH=${CLASSPATH}:${JAR}
done

$JAVA -XX:MaxPermSize=128m -cp ${CLASSPATH} spawnn.gui.SpawnnGui
