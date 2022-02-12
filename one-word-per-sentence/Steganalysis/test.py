def fc(data_tmp):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import random
    random.seed(2021)

    with open('../bert-base-uncased-gen/cover.txt', 'r', encoding='UTF-8') as f:
        cover = f.readlines()

    with open(data_tmp, 'r', encoding='UTF-8') as f:
        stega = f.readlines()

    cover = [t.strip() for t in cover][:10000]
    stega = [t.strip() for t in stega][:10000]
    cover.extend(stega)

    cover_label = [0] * 10000
    stega_label = [1] * 10000

    cover_label.extend(stega_label)

    dic = {'sentence': cover, 'label': cover_label}

    df = pd.DataFrame(dic, columns=['sentence', 'label'])

    train, test = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=2021)
    train, val = train_test_split(train, test_size=0.1, stratify=train['label'], random_state=2021)

    train.to_csv('./data/train.csv')
    val.to_csv('./data/val.csv')
    test.to_csv('./data/test.csv')


