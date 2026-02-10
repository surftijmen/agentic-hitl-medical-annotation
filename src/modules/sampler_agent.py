import pandas as pd


class DataSampler:
    def __init__(
        self,
        notes_path: str,
        adm_path: str,
        seed: int = 12,
    ):
        notes_df = pd.read_parquet(
            notes_path,
            columns=["ROW_ID", "SUBJECT_ID", "HADM_ID", "TEXT"],
        )

        adm_df = pd.read_parquet(
            adm_path,
            columns=["SUBJECT_ID", "HADM_ID", "DIAGNOSIS"],
        )

        self.df = (
            notes_df
            .merge(adm_df, on=["SUBJECT_ID", "HADM_ID"], how="inner")
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        self.cursor = 0

    def sample_batch(self, n: int) -> pd.DataFrame:
        batch = self.df.iloc[self.cursor : self.cursor + n]
        self.cursor += n
        return batch
