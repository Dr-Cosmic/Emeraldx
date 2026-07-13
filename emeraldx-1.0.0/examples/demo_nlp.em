# NLP: stemming, TF-IDF, classification, sentiment, generation, embeddings.
import nlp;
import embed;
import prng;
import strx;
int corpus = 0;
int corpus2 = 0;
int docs = 0;
int dt = 0;
int emodel = 0;
int i = 0;
int labels = 0;
int mk = 0;
int nb = 0;
int text = 0;
# emx-scope-safe

print("== emeraldx: natural language processing ==");
print("porter stems      running->", nlp.porter_stem("running"),
      "  relational->", nlp.porter_stem("relational"),
      "  hopefulness->", nlp.porter_stem("hopefulness"));

docs = [];
labels = [];
docs[0] = nlp.tokenize("win free money claim your prize now");   labels[0] = "spam";
docs[1] = nlp.tokenize("free cash prize win the lottery today"); labels[1] = "spam";
docs[2] = nlp.tokenize("exclusive offer win money act now");     labels[2] = "spam";
docs[3] = nlp.tokenize("agenda for the project meeting");        labels[3] = "ham";
docs[4] = nlp.tokenize("please review the quarterly report");    labels[4] = "ham";
docs[5] = nlp.tokenize("team lunch after the design review");    labels[5] = "ham";
nb = nlp.nb_fit(docs, labels);
print("naive bayes       'claim your free prize' -> ",
      nlp.nb_predict(nb, nlp.tokenize("claim your free prize")));
print("naive bayes       'review the meeting agenda' -> ",
      nlp.nb_predict(nb, nlp.tokenize("review the meeting agenda")));

print("sentiment         'a wonderful, brilliant film' -> ",
      strx.fmt_f(nlp.sentiment("a wonderful, brilliant film"), 2));
print("sentiment         'the food was not good' -> ",
      strx.fmt_f(nlp.sentiment("the food was not good"), 2));

corpus = "the ship sailed over the calm sea. the wind pushed the ship past the rocks. the sea threw waves over the rocks and the shore.";
mk = nlp.markov_fit(corpus, 2);
print("markov(2)         ", nlp.markov_generate(mk, 12, prng.seed_from(5)));

text = "Photogrammetry recovers 3D structure from photographs. Cameras observe the same points from different angles. Triangulation intersects the viewing rays. Modern pipelines add dense matching and meshing. This library implements the two-view core.";
print("summary           ", nlp.summarize(text, 2));

corpus2 = ["the cat chased the mouse", "a dog chased the cat", "the mouse hid from the dog",
           "the bank raised the interest rate", "the market fell on bank news",
           "investors watched the market rate"];
dt = [];
for (i = 0; i < len(corpus2); ++i) { dt[i] = nlp.remove_stopwords(nlp.tokenize(corpus2[i])); }
emodel = embed.fit(dt, 2, 6);
print("embeddings        cat~dog ", strx.fmt_f(embed.similarity(emodel, "cat", "dog"), 2),
      "  bank~market ", strx.fmt_f(embed.similarity(emodel, "bank", "market"), 2),
      "  cat~bank ", strx.fmt_f(embed.similarity(emodel, "cat", "bank"), 2));
