gcloud auth activate-service-account --key-file $1 2> /dev/null || echo 'error, could not connect to service account'
working_folder=$2
bucket=$3
sudo gsutil rm -r gs://$bucket/* 2> /dev/null || echo 'error, the bucket may be empty'
sudo gsutil cp -r $working_folder gs://$bucket 2> /dev/null || echo 'unable to copy files'
