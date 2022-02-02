from pathlib import Path


def init(cfg):
    cfg.competition_path = Path('/kaggle/input/sartorius-cell-instance-segmentation')
    if 'colab' in cfg.tags:
        cfg.competition_path = Path('/content/gdrive/MyDrive/sartorius/sartorius-cell-instance-segmentation')
    cfg.image_root = cfg.competition_path / 'train'
    cfg.meta_csv = cfg.competition_path / 'train.csv'


def add_image_id(df, cfg):
    df.rename(columns={'id': 'image_id'}, inplace=True)

    # Move to extra func?
    if 'celltype' in cfg.tags:
        gb = df.groupby('image_id')
        grouped_df = gb.head(1).set_index('image_id')
        df = grouped_df.reset_index()
        del gb, grouped_df
    elif 'bbox' in cfg.tags:
        df['annotation'] = gb.annotation.apply(','.join)

    return df


def add_category_id(df, cfg):
    if 'celltype' in cfg.tags:
        return df.rename(columns={'cell_type': 'category_id'})
    return df
