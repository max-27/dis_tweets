import logging
import pandas as pd
import os
from src.util import get_root_path


def main(input_filepath, output_filepath=None, columns=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # columns = [id, keyword, location, text, optional(target)]
    train_data = pd.read_csv(os.path.join(input_filepath, "train.csv"))
    test_data = pd.read_csv(os.path.join(input_filepath, "test.csv"))
    if columns:
        logger.info(f"Only features: {columns} in dataset!")
        test_data = test_data[columns]
        columns.append("target")
        train_data = train_data[columns]

    # preprocessing stuff
    pass
8

if __name__ == '__main__':
    out_path = os.path.join(get_root_path(), "data", "processed")
    in_path = os.path.join(get_root_path(), "data", "raw")
    columns = ["id", "text"]
    main(input_filepath=in_path, output_filepath=out_path, columns=columns)
