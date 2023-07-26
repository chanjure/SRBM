import os
from time import time
import torch
from torchvision import transforms

def check_num_workers(dataset, batch_size, verbose=False):
    """
    check_num_workers
    --------------

    Check the optimal number of workers for the dataloader.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        Dataset object.

    batch_size: int

    Returns
    -------
    num_workers: int
        Optimal number of workers.
    """
    num_time = {}

    for num_workers in range(2, len(os.sched_getaffinity(0))+2, 2):
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_dl = torch.utils.data.DataLoader(dataset, \
                batch_size=batch_size, \
                shuffle=True, \
                drop_last=True, \
                **kwargs)

        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_dl, 0):
                pass
        end = time()

        duration = end - start
        num_time[num_workers] = duration
        
        if verbose:
            print(f'num_workers: {num_workers}, duration: {duration}')

    return min(num_time, key=num_time.get)

def S_1d(field, m):
    """
    S_1d
    ------

    Compute action density for 1 dimensional configuration.

    Parameters
    ----------
    field: torch.Tensor
        Field configurations.
    m: float
        Mass of the field.

    Returns
    -------
    S: torch.Tensor
        Action density.
    """
    n_data = field.shape[0]
    N = field.shape[1]

    s = m**2 * field**2
    s += 2.*field**2
    s -= field*torch.roll(field, shifts=-1, dims=1)
    s -= field*torch.roll(field, shifts=1, dims=1)

    return 0.5*s.sum()/n_data/N

