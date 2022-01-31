def filter_bg_image_ids(bg_image_ids, df):
    "Remove unlabelled duplicate images from `bg_image_ids`"
    n_images_per_study = df.groupby('StudyInstanceUID').image_id.count()
    multi_image_studies = set(n_images_per_study.loc[n_images_per_study > 1].index)
    potential_duplicates = set(df.loc[df.StudyInstanceUID.isin(multi_image_studies), 'image_id'])
    print(f'Excluding {len(potential_duplicates)} potentially unlabelled duplicate BG images')
    bg_image_ids = list(set(bg_image_ids) - potential_duplicates)
    print(f'Safe BG images: {len(bg_image_ids)}')
    return bg_image_ids
