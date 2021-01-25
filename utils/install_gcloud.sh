echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
add-apt-repository ppa:deadsnakes/ppa
apt-get install python3.6
apt-get install apt-transport-https ca-certificates gnupg
apt-get -y update && apt-get install -y google-cloud-sdk
