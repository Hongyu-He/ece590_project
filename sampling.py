import numpy as np


def sample_iid(dataset, num_users):
    """Sample iid data from dataset for each user.

    Args:
        dataset (torch.datasets): train set to sample from
        num_users (int): number of users

    Returns:
        dict_users: dictionary of data index for each user 
            {user_id: set(data_index)}
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def sample_noniid(dataset, num_users):
    """Sample non-iid data from dataset for each user.

    Args:
        dataset ([type]): [description]
        num_users ([type]): [description]
    """
    # TODO implement sample non-iid
    pass