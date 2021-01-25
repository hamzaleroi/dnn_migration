## How to use these files?
1. First install gcloud using: `./install_gcloud.sh`
2. Once it is installed, you can upload files with:
 ```bash
upload_files.sh <Service account JSON File> <Folder to upload> <bucket>
```
Where:
* <Service account JSON File> is the file for the service account having the create object role on GCP
* <Folder to upload> is the folder to upload to GCP
* <bucket> is the destination bucket
 
