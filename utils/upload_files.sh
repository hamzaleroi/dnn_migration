gcloud auth activate-service-account --key-file $1
working_folder=$2
bucket=$3
gsutil rm -r gs://$bucket/*
gsutil cp -r $working_folder gs://$bucket
