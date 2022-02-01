def filter_bg_image_ids(bg_image_ids, df):
    "Remove unlabelled duplicate images from `bg_image_ids`"
    bg_image_ids = set(bg_image_ids)
    n_images_per_study = df.groupby('StudyInstanceUID').image_id.count()
    multi_image_studies = set(n_images_per_study.loc[n_images_per_study > 1].index)
    potential_duplicates = set(df.loc[df.StudyInstanceUID.isin(multi_image_studies), 'image_id'])
    potential_duplicates = potential_duplicates.intersection(bg_image_ids)
    print(f'Excluding {len(potential_duplicates)} / {len(bg_image_ids)} potentially unlabelled duplicate BG images')
    bg_image_ids = list(bg_image_ids - potential_duplicates)
    print(f'Safe BG images: {len(bg_image_ids)}')
    return bg_image_ids


def filter_bg_images(bg_images, df):
    "Remove unlabelled duplicate images from `bg_images`"
    bg_images = set(bg_images)
    n_images_per_study = df.groupby('StudyInstanceUID').image_id.count()
    multi_image_studies = set(n_images_per_study.loc[n_images_per_study > 1].index)
    potential_duplicates = set(df.loc[df.StudyInstanceUID.isin(multi_image_studies), 'image_path'])
    potential_duplicates = potential_duplicates.intersection(bg_images)
    print(f'Excluding {len(potential_duplicates)} / {len(bg_images)} potentially unlabelled duplicate BG images')
    bg_images = list(bg_images - potential_duplicates)
    print(f'Safe BG images: {len(bg_images)}')
    return bg_images
