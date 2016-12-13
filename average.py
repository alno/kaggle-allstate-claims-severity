import pandas as pd

files = [
    '20161212-2301-l3-qr-foldavg-1115.79637-test-foldavg',
    '20161212-2301-l3-qr-foldavg-1115.79637-test-foldavg',
    '20161212-2301-l3-qr-foldavg-1115.79637-test-fulltrain',

    '20161212-1117-l3-qr-foldavg-1115.78410-test-foldavg',
    '20161212-1117-l3-qr-foldavg-1115.78410-test-fulltrain',

    '20161209-2037-l3-qr-1116.11284-test-foldavg',
    '20161209-2037-l3-qr-1116.11284-test-fulltrain',

    '20161211-2243-l3-qr-foldavg-1115.79246-test-foldavg',

    '20161205-2331-l3-qr-1116.35041-test-foldavg',

    '20161203-1654-l3-qr-1116.42746-test-foldavg',

    '20161210-2259-l3-qr-3-1115.89568-test-foldavg',

    '20161130-1040-l3-qr-1116.68777-test',

    '20161202-1115-l3-qr-1116.58519-test',
]


preds = [pd.read_csv('preds/%s.csv' % f, index_col='id') for f in files]

pred = reduce(lambda a, b: a + b, preds) / len(preds)
pred.to_csv('avg-2.csv')

print "Done."
