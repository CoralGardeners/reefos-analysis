import numpy as np
import pandas as pd


def get_species_counts(detections):
    results = [rec.values for table in detections for rec in table.records]
    if len(results) == 0:
        print("ERROR: no detections")
        return None
    df = pd.DataFrame(results).drop(columns=['result', 'table'])
    _df = df[df._field == 'bbox_top_left_x'].groupby('_time')['class'].value_counts()
    cnt_df = _df.unstack().fillna(0).reset_index()
    return cnt_df


def get_fish_health_from_counts(count_df, min_count=100):
    def divide_and_fill_na(vals1, vals2):
        return np.log10(np.divide(vals1, vals2,
                                  out=np.zeros_like(vals1),
                                  where=(vals2 != 0)))

    labels = ['brown_tang', 'butterflyfish', 'fish',
              'parrotfish', 'surgeonfish']

    if count_df is None:
        return None, None
    class_counts = count_df[labels].sum()
    total_fish_counted = sum(class_counts[label] for label in labels)
    if total_fish_counted <= min_count:
        return None, None

    # compute log abundance ratios
    sf_bt = divide_and_fill_na(class_counts['surgeonfish'], class_counts['brown_tang'])
    # sf_pf = divide_and_fill_na(class_counts['surgeonfish'], class_counts['parrotfish'])
    bf_bt = divide_and_fill_na(class_counts['butterflyfish'], class_counts['brown_tang'])
    bf_pf = divide_and_fill_na(class_counts['butterflyfish'], class_counts['parrotfish'])

    # compute cover estimate from fish abundance ratios. MCR Linear regression results:
    # ['Log_Cropper_CC_Ratio', 'Log_Corallivore_CC_Ratio', 'Log_Corallivore_ScrapeExcav_Ratio']
    # Linear model R^2 0.5868659580053202
    # [    0.30485    0.091695      0.3027]
    # -0.8072201982465221
    estimated_log_coral = (0.30 * sf_bt + 0.092 * bf_bt + 0.30 * bf_pf - 0.81)
    # health is linear position in range (0.02, 0.5)
    min_log = np.log10(0.02)
    max_log = np.log10(0.50)
    health = (estimated_log_coral - min_log) / (max_log - min_log)
    # clip and scale to (0, 100)
    health = 100 * np.clip(health, 0, 1)

    return health, (class_counts / class_counts.sum()).to_dict()


def get_fish_health(detections, min_count=100):
    if len(detections) == 0:
        return 0, None
    # get class counts from raw detections
    count_df = get_species_counts(detections)
    return get_fish_health_from_counts(count_df, min_count=min_count)


def get_coral_health():
    return 20


def get_combined_health(components):
    return np.array(list(components.values())).mean()
