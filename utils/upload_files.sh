gcloud auth activate-service-account --key-file $1 > /dev/null|| echo 'error, could not connect to service account'
working_folder=$2
bucket=$3
echo 'deleting files from  the bucket'
sudo gsutil rm -r gs://$bucket/* > /dev/null \
        || echo 'error, the bucket may be empty'
echo "copying the files from $working_folder"
sudo gsutil cp -r $working_folder gs://$bucket >/dev/null \
        || echo 'unable to copy files'


