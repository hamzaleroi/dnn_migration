curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1-6GulWdUGLBm76f9Y1IOFzg6h7vo1y06" > /dev/null 
code="$(awk '/_warning_/ { print $NF }' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=1-6GulWdUGLBm76f9Y1IOFzg6h7vo1y06" -o screw_detection.zip
unzip screw_detection.zip