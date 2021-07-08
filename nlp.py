import nltk
import matplotlib.pyplot as plt

# Setup nltk libs
nltk.download("averaged_perceptron_tagger")


def get_tags(tokens):
    return nltk.tag.pos_tag(tokens)


def plot_tags_freq(tags):
    tags_dict = {}
    for (k,v) in tags:
        if v in tags_dict:
            tags_dict[v] = tags_dict[v] + 1
        else:
            tags_dict[v] = 1

    sorted_dict = dict(sorted(
        tags_dict.items(), 
        key = lambda item: item[1], 
        reverse=True
    ))
    plt.figure(figsize=(15,5))
    fig, ax = plt.subplots()
    
    ax.bar(list(sorted_dict.keys()), sorted_dict.values())
    ax.set_yscale("log")
    
    ax.set_xlabel("POS Tags")
    ax.set_ylabel("log of counts")
    
    plt.tick_params(axis="x", which="major", labelsize=8)
    plt.savefig("results/pos-tag-frequency.png")


