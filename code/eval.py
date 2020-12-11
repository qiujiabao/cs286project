import nltk

with open('reference.txt', 'r') as input1, open('translation.txt', 'r') as input2:
    corpus_score = 0.0
    sentence_score = 0.0
    text1 = input1.read().split('\n')
    text2 = input2.read().split('\n')
    print(len(text1))
    print(len(text2))
    corpus_score = corpus_score + nltk.translate.bleu_score.corpus_bleu(text1, text2, weights=[1])
    for i in range(0, len(text1)):
        sentence_score = sentence_score + nltk.translate.bleu_score.sentence_bleu(text1[i], text2[i], weights=[1])
    print(corpus_score)
    print(sentence_score / len(text1))
