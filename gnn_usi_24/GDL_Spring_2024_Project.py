from data_prep import compute_diff_ts, minmax_bound_scaler,unit_centered_scaler, check_nans
import pandas as pd
import pathlib
import os
import torch
from tsl.ops.connectivity import adj_to_edge_index

from tsl.data import SpatioTemporalDataset, BatchMapItem, SynchMode, BatchMap

import numpy as np


DATASET_NAME = "../merged_raw_dataset.parquet"

stocknet_df = pd.read_parquet(DATASET_NAME)



stocknet_diff, invert_minmax, invert_df = unit_centered_scaler(
                    compute_diff_ts(stocknet_df)
                )
stocknet_diff.dropna(axis=1,inplace=True)
STOCKTABLES_PATH = pathlib.Path("../stocknet-dataset/StockTable")
    
stock_sectors_df = pd.read_table(STOCKTABLES_PATH)


stock_sectors_df["Symbol"] =stock_sectors_df["Symbol"].str.lstrip("$")


stock_sectors_df = stock_sectors_df.set_index("Symbol")
stocknet_diff_column_order = stocknet_diff.columns.get_level_values(0).unique()
stock_sectors_df = stock_sectors_df.reindex(stocknet_diff_column_order)

adjacency_rows = []
for stockname_to_sector in stock_sectors_df.itertuples():
  is_sector_equal_row = stockname_to_sector.Sector == stock_sectors_df["Sector"]
  adjacency_rows.append(is_sector_equal_row.astype(int).values)

adjacency_matrix_by_sector = np.stack(adjacency_rows)

edge_index_sector = adj_to_edge_index(adjacency_matrix_by_sector)



class StocknetDataset(SpatioTemporalDataset):
    

    def __init__(self, target, *args, **kwargs):
        self.shit = torch.tensor(0)
        super(StocknetDataset, self).__init__(target=target, *args, **kwargs)

        


# In[99]:

# test_item =BatchMapItem('shit',
#                                             SynchMode.HORIZON,
#                                             preprocess=False,
#                                             cat_dim=None,
#                                             pattern='t n',
#                                             shape=stocknet_diff.shape)
#
# test_map = BatchMap(y=test_item)
def make_binary_classification(data):
    adjusted_close = data.y[:,:,-2]
    adjusted_close_rises = (adjusted_close > 0)
    data.y = adjusted_close_rises.to(torch.int)
    return data


print(stocknet_diff)
test_df = StocknetDataset(target=stocknet_diff,transform=make_binary_classification,
                                     connectivity=edge_index_sector,
                                      horizon=1,
                                      window=12,
                                      stride=1)

print(test_df[0].y)