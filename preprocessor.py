def create_vocab(data):
    vocab = {}
    for raw in data:
        for word in raw.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab
