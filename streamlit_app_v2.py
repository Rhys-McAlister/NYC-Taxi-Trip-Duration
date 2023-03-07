from dataclasses import dataclass
import pandas as pd

@dataclass
class IngestDataFromTlc:
    year: int
    month: int
    colour: str
    data: pd.DataFrame = None