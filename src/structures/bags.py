import numpy as np
from torch.utils.data import Dataset


class BagDataset(Dataset):
    """Utility class to work with bagged data. Every bag is assumed to have the
    same dimensionality and a given aggregate label

    # TODO :
        - rename attribute
        - allow use of torch tensors
        - allow individuals to be fed as numpy array with bags sizes

    Args:
        bags (list[np.ndarray]): list of bags of features with size [(bag_size, dim)]
        labels (list, np.ndarray): labels associated to each bag with size (Σ bag_sizes)
        stacked (bool): if True, bags are stacked into single numpy array
        copy (bool): if True, inputs are copied
        meta (dict): any additional metadata

    Attributes:
        bags (list[np.ndarray], np.ndarray): if stacked==True, bags features are stacked
            into single numpy array
        labels (list, np.ndarray): labels associated to each bag with size (Σ bag_sizes)
        stacked (bool): if True, bags are stacked into single numpy array
        copy (bool): if True, inputs are copied
        dim (int): bags features dimensionality
        n_bags (int): number of bags
        dtype (type): data type of bag features
        sizes (list[int]): number of individuals per bag
        n_individuals (int): total number of individuals
        meta (dict): any additional metadata
    """
    def __init__(self, bags, labels, stack=True, copy=True, meta=None):
        # Run sanity check on inputs
        self._assert_valid(bags=bags, labels=labels)

        # Set metadata attributes
        self.dim = bags[0].shape[-1]
        self.n_bags = len(bags)
        self.dtype = bags[0].dtype
        self.sizes = list(map(len, bags))
        self.n_individuals = sum(self.sizes)
        self.meta = meta if meta else {}

        # Set data handling attributes
        self.copy = copy
        self.stacked = False

        # Set data attributes
        self.bags = bags
        self.labels = labels
        if stack:
            self.stack()

    def stack(self, inplace=True):
        """Stacks bags into single numpy array

        Args:
            inplace (bool): if True, updates bags instance attribute
        """
        bags = self.bags
        if not self.stacked:
            bags = np.vstack(bags)

        if inplace:
            self.bags = bags
            self.stacked = True
        else:
            return bags

    def unstack(self, inplace=True):
        """Splits stacked bags into list of numpy arrays

        Args:
            inplace (bool): if True, updates bags instance attribute
        """
        bags = self.bags
        if self.stacked:
            bags = np.split(self.bags, indices_or_sections=self._cumsizes[1:-1])

        if inplace:
            self.bags = bags
            self.stacked = False
        else:
            return bags

    def _assert_valid(self, bags, labels):
        """Sanity check for bags and labels provided

        Args:
            bags (list[np.ndarray]): list of bags of features
            labels (list, np.ndarray): list of bag labels
        """
        # Make sure we have pairs of bag and label
        if len(bags) != len(labels):
            raise ValueError(f"Mismatch between number of bags ({len(bags)}) and labels ({len(labels)})")

        # Make sure bags are provided as a list
        if not isinstance(bags, list):
            raise TypeError(f"Expected bags concatenated as list, got {type(bags)}")

        # Make sure each bag is a numpy array
        bags_types = set(map(lambda x: isinstance(x, np.ndarray), bags))
        is_numpy_array = bags_types.pop()
        if not is_numpy_array or bags_types:
            raise TypeError(f"Expected bags as numpy arrays, got types {bags_types}")

        # Make sure all bags have the same dimensionality
        dims = set([x.ndim for x in bags])
        if len(dims) > 1:
            raise ValueError(f"Mismatch between number of dimensions in bag features, found dimensions {(dims)}")

    def __getitem__(self, idx):
        """Retrieves bags and targets corresponding to specified index

        Args:
            idx (int, slice)

        Returns:
            type: np.ndarray, list[np.ndarray]
        """
        if self.stacked:
            # If single index, return individual bag
            if isinstance(idx, int):
                bags = self.bags[self._cumsizes[idx]:self._cumsizes[idx + 1]]
            # If multiple indices, return list of selected bags
            elif isinstance(idx, slice):
                start = idx.start if idx.start else 0
                stop = idx.stop if idx.stop else len(self)
                step = idx.step if idx.step else 1
                bags = [self.bags[self._cumsizes[i]:self._cumsizes[i + 1]] for i in range(start, stop, step)]
            else:
                raise TypeError(f"Expected slice type index, got {type(idx)}")
        else:
            # When bags aren't stacked, we can use list slicing
            bags = self.bags[idx]

        # Get corresponding labels
        labels = self.labels[idx]
        return bags, labels

    def __eq__(self, bag_dataset):
        raise NotImplementedError

    def __ne__(self, bag_dataset):
        raise NotImplementedError

    def __add__(self, bag_dataset):
        # Unstack datasets
        self_bags = self.unstack(inplace=False)
        other_bags = bag_dataset.unstack(inplace=False)

        # Concatenate bags and labels
        concatenated_bags = self_bags + other_bags
        concatenated_labels = np.concatenate([self.labels, bag_dataset.labels])

        # Encapsulate into new bag dataset
        concatenated_dataset = BagDataset(bags=concatenated_bags,
                                          labels=concatenated_labels,
                                          stack=self.stacked)
        return concatenated_dataset

    def __radd__(self, bag_dataset):
        return bag_dataset.__add__(self)

    def centroids(self):
        bags = self.unstack(inplace=False)
        centroids = [np.mean(x, axis=0) for x in bags]
        return centroids

    def __len__(self):
        return self.n_bags

    def __repr__(self):
        output = f"Number of bags : {self.n_bags} \n"
        output += f"Dimensionality : {self.dim} \n"
        output += f"Number of individuals : {self.n_individuals} \n"
        output += f"Data type : {self.dtype}"
        return output

    @property
    def stacked(self):
        return self._stacked

    @property
    def copy(self):
        return self._copy

    @property
    def dim(self):
        return self._dim

    @property
    def n_bags(self):
        return self._n_bags

    @property
    def sizes(self):
        return self._sizes

    @property
    def n_individuals(self):
        return self._n_individuals

    @property
    def dtype(self):
        return self._dtype

    @property
    def bags(self):
        return self._bags

    @property
    def labels(self):
        return self._labels

    @stacked.setter
    def stacked(self, stacked):
        self._stacked = stacked

    @copy.setter
    def copy(self, copy):
        self._copy = copy

    @dim.setter
    def dim(self, dim):
        self._dim = dim

    @n_bags.setter
    def n_bags(self, n_bags):
        self._n_bags = n_bags

    @sizes.setter
    def sizes(self, sizes):
        self._sizes = sizes
        self._cumsizes = np.cumsum([0] + sizes)

    @n_individuals.setter
    def n_individuals(self, n_individuals):
        self._n_individuals = n_individuals

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @bags.setter
    def bags(self, bags):
        if self.copy:
            if isinstance(bags, list):
                self._bags = [x.copy() for x in bags]
            elif isinstance(bags, np.ndarray):
                self._bags = bags.copy()
        else:
            self._bags = bags

    @labels.setter
    def labels(self, labels):
        self._labels = np.asarray(labels).copy()
