from torch.utils.data import DataLoader

from irl.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        min_ped=0)                              # added this line to include frames with 1 ped as well (default >1)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        # num_workers=0,
        collate_fn=seq_collate)
    return dset, loader
