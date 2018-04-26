import numpy as np

class DataSet(object):

    def __init__(self, images, labels, cls, class_ls):
        """ initialize the DataSet class
            Args: 
                images (numpy ndarray): 2-D flatted images numpy array
                labels (numpy ndarray): 2-D one hot labels numpy array
                cls (numpy ndarray): 1-D index of the labels (0 : class_num)
                class_ls (list): list of classes
        """

        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._cls = cls
        self._label_texts = np.array([class_ls[idx] for idx in cls])

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def label_texts(self):
        return self._label_texts

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """
            Return the next `batch_size` examples from this data set.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # we already did the shuffle at the preprocessing stage

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._cls[start:end]

