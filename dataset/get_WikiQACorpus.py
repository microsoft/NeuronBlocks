# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import urllib.request
import zipfile

data_path = 'https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip'
data_file = 'WikiQACorpus.zip'
print("Downloading and extracting %s..." % data_file)
urllib.request.urlretrieve(data_path, data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall()
os.remove(data_file)
print("Completed!")
