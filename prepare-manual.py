import pandas as pd

from util import Dataset

for name in ['train', 'test']:
    print "Processing %s..." % name

    idx = Dataset.load_part(name, 'id')

    # Load parts
    numeric = pd.DataFrame(Dataset.load_part(name, 'numeric'), columns=Dataset.get_part_features('numeric_lin'), index=idx)
    numeric_lin = pd.DataFrame(Dataset.load_part(name, 'numeric_lin'), columns=Dataset.get_part_features('numeric_lin'), index=idx)

    # Build features
    df = pd.DataFrame(index=idx)
    #df['cont14'] = numeric['cont14']
    df['cont_1_9_diff'] = numeric_lin['cont9'] - numeric_lin['cont1']

    # Save column names
    if name == 'train':
        Dataset.save_part_features('manual', list(df.columns))

    Dataset(manual=df.values).save(name)

print "Done."
