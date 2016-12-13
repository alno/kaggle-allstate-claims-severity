import sys

from sklearn.metrics import mean_absolute_error
from util import Dataset, load_prediction


loss = Dataset.load_part('train', 'loss')
pred = load_prediction('train', sys.argv[1])

step = 0.01
cur = 1.02

while step >= 1e-3:
    best_p = cur
    best_mae = 1e9

    for i in range(-9, 10):
        p = cur + i * step

        corr_pred = (pred ** p) / (pred.mean() ** (p - 1))
        corr_mae = mean_absolute_error(loss, corr_pred)

        print "  Corr pow %.4f MAE: %.5f " % (p, corr_mae)

        if corr_mae < best_mae:
            best_p = p
            best_mae = corr_mae

    cur = best_p
    step /= 10

    print "Best corr pow %.4f MAE: %.5f " % (best_p, best_mae)

print "Done."
