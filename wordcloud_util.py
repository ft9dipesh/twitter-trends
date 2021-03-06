import math

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def get_sorted_phrasecounts(phrases, frequency, reverse=True):
    d = dict(zip(phrases, frequency))
    d = {k: 1 if math.isnan(v) else v for k, v in d.items()}

    return dict(sorted(d.items(), key = lambda item: item[1], reverse=reverse))

def generate_wordcloud(phrases, frequency, mask):
    d_sorted = get_sorted_phrasecounts(phrases, frequency)

    wordcloud = WordCloud(
        font_path="./fonts/KlokanTechNotoSans-Regular.ttf", 
        width=800, 
        height=800, 
        collocations=False,
        mask=mask,
        #background_color="white",
        #color_func=lambda *args, **kwargs: "black",
    ).generate_from_frequencies(d_sorted)

    fig = plt.figure(figsize=(16,16))
    fig.patch.set_facecolor("black")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig("results/wordcloud.png")


