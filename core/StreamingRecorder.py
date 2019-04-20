# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np

class StreamingRecorder():
    def __init__(self, names):
        """

        Args:
            names:  ['prediction', ... ]
        """
        self.__names = names
        self.__operators = dict()
        self.__recorder = dict()
        for name in names:
            self.__recorder[name] = []

    def record(self, name, values, keep_dim=False):
        """ insert a col of multiple values

        Args:
            name:
            values:

        Returns:

        """
        if isinstance(values, list) or isinstance(values, np.ndarray):
            if keep_dim is False:
                self.__recorder[name].extend(values)
            else:
                self.__recorder[name].append(values)
        else:
            self.__recorder[name].append(values)

    def record_one_row(self, values, keep_dim=False):
        """ insert a whole row

        Args:
            values: [col1, col2, col3, ...], each element can be either a list or a single number

        Returns:

        """
        assert len(self.__names) == len(values)
        for name, value in zip(self.__names, values):
            self.record(name, value, keep_dim)

    def get(self, name, operator=None):
        """

        Args:
            name:
            operator: has the same shape with names, supported operations:
                    None or 'origin': return the original values
                    'mean': return mean of the values
                    'sum': return sum of the values
                    'min': return min of the values
                    'max': return max of the values
                    'distribution': return 0%, 10%, 20%, ..., 90%, 100% of values, from min to max

        Returns:

        """

        if operator is None or operator == 'origin':
            return self.__recorder[name]
        elif operator == 'mean':
            return np.mean(self.__recorder[name])
        elif operator == 'sum':
            return np.sum(self.__recorder[name])
        elif operator == 'min':
            return np.min(self.__recorder[name])
        elif operator == 'max':
            return np.max(self.__recorder[name])
        elif operator == 'distribution':
            data_sorted = np.sort(self.__recorder[name])
            distribution = []
            for i in np.linspace(0, 1, 11):
                if i != 1:
                    distribution.append(data_sorted[int(i * len(data_sorted))])
                else:
                    distribution.append(data_sorted[-1])
            return distribution

    def clear_records(self, name=None):
        if name is None:
            for name in self.__names:
                self.__recorder[name] = []
        else:
            self.__recorder[name] = []




if __name__ == "__main__":
    streaming_recorder = StreamingRecorder(['prediction'])
    streaming_recorder.record('prediction', [1, 2, 3])
    streaming_recorder.record('prediction', [4, 5, 6])
    print(streaming_recorder.get('prediction', 'origin'))
    print(streaming_recorder.get('prediction', 'distribution'))

