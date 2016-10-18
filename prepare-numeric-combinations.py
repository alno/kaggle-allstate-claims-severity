import pandas as pd

from util import Dataset

for name in ['train', 'test']:
    print "Processing %s..." % name

    num = pd.DataFrame(Dataset.load_part(name, 'numeric'), columns=Dataset.get_part_features('numeric'))
    df = pd.DataFrame(index=num.index)

    df['diff_1_6'] = num['cont1'] - num['cont6']
    df['diff_1_9'] = num['cont1'] - num['cont9']
    df['diff_1_10'] = num['cont1'] - num['cont10']
    df['diff_6_9'] = num['cont6'] - num['cont9']
    df['diff_6_10'] = num['cont6'] - num['cont10']
    df['diff_6_11'] = num['cont6'] - num['cont11']
    df['diff_6_12'] = num['cont6'] - num['cont12']
    df['diff_6_13'] = num['cont6'] - num['cont13']
    df['diff_7_11'] = num['cont7'] - num['cont11']
    df['diff_7_12'] = num['cont7'] - num['cont12']
    df['diff_11_12'] = num['cont11'] - num['cont12']

    if name == 'train':
        Dataset.save_part_features('numeric_combinations', list(df.columns))

    Dataset(numeric_combinations=df.values).save(name)

print "Done."
