gcloud auth activate-service-account --key-file $1 || exit
working_folder=$2
bucket=$3
sudo gsutil rm -r gs://$bucket/*
sudo gsutil cp -r $working_folder gs://$bucket
