# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import urllib.request
import zipfile

data_path = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8'
data_file = 'SST-2.zip'
print("Downloading and extracting %s..." % data_file)
urllib.request.urlretrieve(data_path, data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall()
os.remove(data_file)
print("Completed!")
