# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import ftfy


class EnglishTextPreprocessor(object):
    def __init__(self, DBC2SBC=True, unicode_fix=True):
        self.__DBC2SBC = DBC2SBC
        self.__unicode_fix = unicode_fix

    def preprocess(self, string):
        if self.__unicode_fix:
            string = ftfy.fix_text(string)
        if self.__DBC2SBC:
            string = self.DBC2SBC(string)
        return string

    def DBC2SBC(self, ustring):
        """ DBC characters to SBC

        Args:
            ustring:

        Returns:

        """
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    def SBC2DBC(ustring):
        """ SBC to DBC

        Returns:

        """
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x0020:
                inside_code = 0x3000
            else:
                if not (0x0021 <= inside_code and inside_code <= 0x7e):
                    rstring += uchar
                    continue
                inside_code += 0xfee0
            rstring += chr(inside_code)
        return rstring