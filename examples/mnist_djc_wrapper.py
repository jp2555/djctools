from torch.utils.data import Dataset

class MNIST_DJC(Dataset):
    def __init__(self, mnist_dataset):
        self.ds = mnist_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]

        # djctools trainer expects dict-based batches
        return {
            "inputs": x,     # image
            "labels": y,     # classification label
            "truth": y,      # LossModule expects truth here
        }