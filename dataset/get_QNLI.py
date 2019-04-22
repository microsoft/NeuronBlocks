# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import urllib.request
import zipfile

data_path = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601'
data_file = 'QNLI.zip'
print("Downloading and extracting %s..." % data_file)
urllib.request.urlretrieve(data_path, data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall()
os.remove(data_file)
print("Completed!")
