from .utils import load_dataset as _load_dataset

# Load Stocks datasets
bitcoin_data = _load_dataset('bitcoin', 'Date')
