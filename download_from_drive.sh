# The file containing all the IDs from drive
filename=$1
while read line; do
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$line" > /dev/null 
code="$(awk '/_warning_/ { print $NF }' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=$line" -o screw_detection.tar
tar -xvf screw_detection.tar
done <  $filename