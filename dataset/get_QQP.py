# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import urllib.request
import zipfile

data_path = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5'
data_file = 'QQP.zip'
print("Downloading and extracting %s..." % data_file)
urllib.request.urlretrieve(data_path, data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall()
os.remove(data_file)
print("Completed!")
