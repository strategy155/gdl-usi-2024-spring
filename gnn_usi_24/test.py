import torch
from tsl.data import SpatioTemporalDataset
import pandas as pd
import pathlib


class StocknetDataset(SpatioTemporalDataset):

    def __init__(self, target, *args, **kwargs):
        self.shit = torch.tensor(0)
        super(StocknetDataset, self).__init__(target=target, *args, **kwargs)


STOCKTABLES_PATH = pathlib.Path("stocknet-dataset/StockTable")

stock_sectors_df = pd.read_table(STOCKTABLES_PATH)
stock_sectors_df.head()

test_df = StocknetDataset(target=stocknet_diff,target_map = {'y': ["target", "shit"]},
                                     connectivity=edge_index_sector,
                                      horizon=1,
                                      window=12,
                                      stride=1)
print(test_df.target_map['y'].__dict__)