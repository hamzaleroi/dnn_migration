#!/bin/bash
# The file containing all the IDs from drive
filename=$1
dir=$(pwd)
while IFS=, read -r id name
do
    echo "downloading $id $name"
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$id" > /dev/null 
    code="$(awk '/_warning_/ { print $NF }' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=$id" -o $name.tar
    tar -xvf  $name.tar --strip-components=1
done <  $filename