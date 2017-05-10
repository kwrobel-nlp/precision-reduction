import csv
import numpy as np
import collections
import sys

AVERAGING = 'macro' # macro or micro
PRECISIONS = [16, 8, 7, 6, 5, 4, 3, 2, 1]

path = sys.argv[1]


f = open(path, 'r')
reader = csv.reader(f)


s = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))))

for row in reader:
    corpus = row[0]
    model_name = row[1]
    clf = row[2]
    fold = row[3]
    micro_precision = row[4]
    macro_precision = row[5]
    macro_recall = row[6]
    macro_f1 = row[7]
    weighted_f1 = row[8]
    train_time = float(row[9])
    train_time_clf = float(row[10])
    test_time = float(row[11])
    test_time_clf = float(row[12])

    if AVERAGING=='macro':
        acc=macro_f1
    elif AVERAGING=='micro':
        acc=micro_precision

    acc=float(acc)
    mn = model_name.split('_')
    model = mn[0]

    r='non'
    for x in mn[1:]:
        if x=='':
            pass
        elif x[0]=='R':
            r=int(x[2:-1])
            model+='_'+x[0]
        else:
            model+='_'+x
            svd=x

    if r=='bez':
        if model=='TFIDF':
            s[corpus]['TFIDF_R'][clf][r].append(acc)
        else:
            s[corpus]['TFIDF_R_'+svd][clf][r].append(acc)
            s[corpus]['TFIDF_'+svd+'_R'][clf][r].append(acc)

    else:
        s[corpus][model][clf][r].append(acc)



for corpus in s:
    print('<h1>%s</h1>' % corpus)
    print('<table>')
    nagl = ['model','clf','non',16,8,7,6,5,4,3,2,1]

    print('<tr>%s</tr>' % (''.join(['<td>'+str(x)+'</td>' for x in nagl]) ))

    for model in sorted(s[corpus]):
        for clf in sorted(s[corpus][model]):
            print('<tr><td>%s</td><td>%s</td>' %(model, clf))

            for r in ['non',16,8,7,6,5,4,3,2,1]:
                if r in s[corpus][model][clf]:
                    mean = np.mean(s[corpus][model][clf][r])
                else:
                    mean=-1.0
                print('<td title="%s">%f</td>' % (r, mean))

            print('</tr>')

    print('</table>')