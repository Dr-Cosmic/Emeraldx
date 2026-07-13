# =============================================================================
# nlp.em - natural language processing
#
#   tokenize, sentences, remove_stopwords, ngrams
#   porter_stem                       the classic Porter (1980) stemmer
#   build_vocab, bow_vector, tfidf_vectors, cosine_bow
#   nb_fit / nb_predict               multinomial naive Bayes classifier
#   sentiment                         lexicon + negation scoring
#   markov_fit / markov_generate      order-k word-level text generator
#   summarize                         extractive TF-scored summarizer
#   levenshtein                       edit distance
# =============================================================================

import mathx;
import arrayx;
import strx;
import dictx;
import prng;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Lowercased alphanumeric word tokens.
fn tokenize(text) {
    int apos = 0; int c = 0; int cur = 0; int i = 0; int low = 0; int out = 0;
    out = [];
    cur = "";
    low = strx.to_lower(text);
    for (i = 0; i < len(low); ++i) {
        c = low[i];
        apos = "'";
        if (strx.is_alnum(c) || c == apos[0]) {
            cur = cur + strx.char_str(c);
        } else {
            if (len(cur) > 0) { out[len(out)] = cur; cur = ""; }
        }
    }
    if (len(cur) > 0) { out[len(out)] = cur; }
    return out;
}

# Split text into sentences on . ! ? boundaries.
fn sentences(text) {
    int c = 0; int cur = 0; int i = 0; int out = 0; int t = 0;
    out = [];
    cur = "";
    for (i = 0; i < len(text); ++i) {
        c = text[i];
        cur = cur + strx.char_str(c);
        if (c == '.' || c == '!' || c == '?') {
            t = strx.trim(cur);
            if (len(t) > 1) { out[len(out)] = t; }
            cur = "";
        }
    }
    t = strx.trim(cur);
    if (len(t) > 1) { out[len(out)] = t; }
    return out;
}

fn stopword_list() {
    return ["a", "an", "the", "and", "or", "but", "if", "then", "so", "of",
            "to", "in", "on", "at", "by", "for", "with", "from", "as", "is",
            "are", "was", "were", "be", "been", "being", "am", "it", "its",
            "this", "that", "these", "those", "i", "you", "he", "she", "we",
            "they", "them", "his", "her", "their", "my", "your", "our", "me",
            "him", "us", "do", "does", "did", "not", "no", "have", "has",
            "had", "will", "would", "can", "could", "should", "may", "might",
            "there", "here", "what", "which", "who", "when", "where", "how",
            "all", "each", "more", "most", "some", "such", "only", "own",
            "same", "than", "too", "very", "just", "about", "into", "over",
            "under", "again", "once", "up", "down", "out", "off"];
}

fn remove_stopwords(tokens) {
    int out = 0; int stop = 0;
    stop = dictx.dnew();
    for w in stopword_list() { stop = dictx.dset(stop, w, 1); }
    out = [];
    for w in tokens {
        if (dictx.dhas(stop, w) == false) { out[len(out)] = w; }
    }
    return out;
}

# n-grams of a token array, joined with '_'.
fn ngrams(tokens, n) {
    int g = 0; int i = 0; int k = 0; int out = 0;
    out = [];
    for (i = 0; i + n <= len(tokens); ++i) {
        g = tokens[i];
        for (k = 1; k < n; ++k) { g = g + "_" + tokens[i + k]; }
        out[len(out)] = g;
    }
    return out;
}

# ---------------------------------------------------------------------------
# Porter stemmer (Porter, 1980). Operates on lowercase words.
# ---------------------------------------------------------------------------

fn p_is_cons(w, i) {
    int c = 0;
    c = w[i];
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') { return false; }
    if (c == 'y') {
        if (i == 0) { return true; }
        return p_is_cons(w, i - 1) == false;
    }
    return true;
}

# m = number of VC sequences in the word.
fn p_measure(w) {
    int i = 0; int m = 0; int n = 0;
    n = len(w);
    m = 0;
    i = 0;
    # skip initial consonants
    while (i < n && p_is_cons(w, i)) { i = i + 1; }
    while (i < n) {
        # in a vowel run
        while (i < n && p_is_cons(w, i) == false) { i = i + 1; }
        if (i >= n) { break; }
        m = m + 1;
        while (i < n && p_is_cons(w, i)) { i = i + 1; }
    }
    return m;
}

fn p_has_vowel(w) {
    int i = 0;
    for (i = 0; i < len(w); ++i) {
        if (p_is_cons(w, i) == false) { return true; }
    }
    return false;
}

fn p_ends_double_cons(w) {
    int n = 0;
    n = len(w);
    if (n < 2) { return false; }
    if (w[n - 1] != w[n - 2]) { return false; }
    return p_is_cons(w, n - 1);
}

# *o: stem ends cvc where the final c is not w, x or y.
fn p_cvc(w) {
    int c = 0; int n = 0;
    n = len(w);
    if (n < 3) { return false; }
    if (p_is_cons(w, n - 3) == false) { return false; }
    if (p_is_cons(w, n - 2)) { return false; }
    if (p_is_cons(w, n - 1) == false) { return false; }
    c = w[n - 1];
    if (c == 'w' || c == 'x' || c == 'y') { return false; }
    return true;
}

fn p_chop(w, k) { return strx.substr(w, 0, len(w) - k); }

# replace suffix `suf` with `rep` if measure of the remaining stem > m0.
# Returns [newWord, applied].
fn p_rule(w, suf, rep, m0) {
    int stem = 0;
    if (strx.ends_with(w, suf) == false) { return [w, false]; }
    stem = p_chop(w, len(suf));
    if (p_measure(stem) > m0) { return [stem + rep, true]; }
    return [w, true];       # suffix matched; rule consumed even if not applied
}

fn porter_stem(word) {
    int c = 0; int done4 = 0; int i = 0; int m = 0; int r = 0; int reps2 = 0; int reps3 = 0; int stem = 0; int step1bExtra = 0; int sufs2 = 0; int sufs3 = 0; int sufs4 = 0; int w = 0;
    w = strx.to_lower(word);
    if (len(w) <= 2) { return w; }

    # ---- step 1a
    if (strx.ends_with(w, "sses")) { w = p_chop(w, 2); }
    elif (strx.ends_with(w, "ies")) { w = p_chop(w, 2); }
    elif (strx.ends_with(w, "ss")) { }
    elif (strx.ends_with(w, "s")) { w = p_chop(w, 1); }

    # ---- step 1b
    step1bExtra = false;
    if (strx.ends_with(w, "eed")) {
        stem = p_chop(w, 3);
        if (p_measure(stem) > 0) { w = stem + "ee"; }
    } elif (strx.ends_with(w, "ed")) {
        stem = p_chop(w, 2);
        if (p_has_vowel(stem)) { w = stem; step1bExtra = true; }
    } elif (strx.ends_with(w, "ing")) {
        stem = p_chop(w, 3);
        if (p_has_vowel(stem)) { w = stem; step1bExtra = true; }
    }
    if (step1bExtra) {
        if (strx.ends_with(w, "at") || strx.ends_with(w, "bl") || strx.ends_with(w, "iz")) {
            w = w + "e";
        } elif (p_ends_double_cons(w)) {
            c = w[len(w) - 1];
            if (c != 'l' && c != 's' && c != 'z') { w = p_chop(w, 1); }
        } elif (p_measure(w) == 1 && p_cvc(w)) {
            w = w + "e";
        }
    }

    # ---- step 1c
    if (strx.ends_with(w, "y")) {
        stem = p_chop(w, 1);
        if (p_has_vowel(stem)) { w = stem + "i"; }
    }

    # ---- step 2 (m > 0 suffix swaps)
    sufs2 = ["ational", "tional", "enci", "anci", "izer", "abli", "alli",
             "entli", "eli", "ousli", "ization", "ation", "ator", "alism",
             "iveness", "fulness", "ousness", "aliti", "iviti", "biliti"];
    reps2 = ["ate", "tion", "ence", "ance", "ize", "able", "al",
             "ent", "e", "ous", "ize", "ate", "ate", "al",
             "ive", "ful", "ous", "al", "ive", "ble"];
    for (i = 0; i < len(sufs2); ++i) {
        if (strx.ends_with(w, sufs2[i])) {
            r = p_rule(w, sufs2[i], reps2[i], 0);
            w = r[0];
            break;
        }
    }

    # ---- step 3
    sufs3 = ["icate", "ative", "alize", "iciti", "ical", "ful", "ness"];
    reps3 = ["ic", "", "al", "ic", "ic", "", ""];
    for (i = 0; i < len(sufs3); ++i) {
        if (strx.ends_with(w, sufs3[i])) {
            r = p_rule(w, sufs3[i], reps3[i], 0);
            w = r[0];
            break;
        }
    }

    # ---- step 4 (m > 1 deletions)
    sufs4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant",
             "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous",
             "ive", "ize"];
    done4 = false;
    for (i = 0; i < len(sufs4); ++i) {
        if (done4 == false && strx.ends_with(w, sufs4[i])) {
            stem = p_chop(w, len(sufs4[i]));
            if (p_measure(stem) > 1) { w = stem; }
            done4 = true;
        }
    }
    if (done4 == false && (strx.ends_with(w, "ion"))) {
        stem = p_chop(w, 3);
        if (p_measure(stem) > 1 && len(stem) > 0) {
            c = stem[len(stem) - 1];
            if (c == 's' || c == 't') { w = stem; }
        }
    }

    # ---- step 5a
    if (strx.ends_with(w, "e")) {
        stem = p_chop(w, 1);
        m = p_measure(stem);
        if (m > 1) { w = stem; }
        elif (m == 1 && p_cvc(stem) == false) { w = stem; }
    }

    # ---- step 5b
    if (p_measure(w) > 1 && p_ends_double_cons(w) && strx.ends_with(w, "l")) {
        w = p_chop(w, 1);
    }
    return w;
}

fn stem_tokens(tokens) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(tokens); ++i) { out[i] = porter_stem(tokens[i]); }
    return out;
}

# ---------------------------------------------------------------------------
# Vector space: vocabulary, bag-of-words, TF-IDF
# ---------------------------------------------------------------------------

# vocab = [wordsArray, indexDict]
fn build_vocab(docsTokens) {
    int idx = 0; int words = 0;
    words = [];
    idx = dictx.dnew();
    for doc in docsTokens {
        for w in doc {
            if (dictx.dhas(idx, w) == false) {
                idx = dictx.dset(idx, w, len(words));
                words[len(words)] = w;
            }
        }
    }
    return [words, idx];
}

fn bow_vector(tokens, vocab) {
    int idx = 0; int k = 0; int v = 0;
    idx = vocab[1];
    v = arrayx.zerosf(len(vocab[0]));
    for w in tokens {
        if (dictx.dhas(idx, w)) {
            k = dictx.dget(idx, w, 0);
            v[k] = v[k] + 1.0;
        }
    }
    return v;
}

# TF-IDF matrix for a token-doc list. Returns [vectors, vocab, idf].
fn tfidf_vectors(docsTokens) {
    int d = 0; int df = 0; int idf = 0; int k = 0; int nDocs = 0; int nWords = 0; int out = 0; int raw = 0; int total = 0; int v = 0; int vecs = 0; int vocab = 0;
    vocab = build_vocab(docsTokens);
    nDocs = len(docsTokens);
    nWords = len(vocab[0]);
    df = arrayx.zerosf(nWords);
    raw = [];
    for (d = 0; d < nDocs; ++d) {
        v = bow_vector(docsTokens[d], vocab);
        raw[d] = v;
        for (k = 0; k < nWords; ++k) {
            if (v[k] > 0.0) { df[k] = df[k] + 1.0; }
        }
    }
    idf = [];
    for (k = 0; k < nWords; ++k) {
        idf[k] = ln((nDocs + 1.0) / (df[k] + 1.0)) + 1.0;
    }
    vecs = [];
    for (d = 0; d < nDocs; ++d) {
        v = raw[d];
        total = arrayx.sumv(v);
        if (total < 1.0) { total = 1.0; }
        out = [];
        for (k = 0; k < nWords; ++k) { out[k] = (v[k] / total) * idf[k]; }
        vecs[d] = out;
    }
    return [vecs, vocab, idf];
}

fn cosine_bow(a, b) {
    int dot = 0; int i = 0; int na = 0; int nb = 0;
    dot = 0.0;
    na = 0.0;
    nb = 0.0;
    for (i = 0; i < len(a); ++i) {
        dot = dot + a[i] * b[i];
        na = na + a[i] * a[i];
        nb = nb + b[i] * b[i];
    }
    if (na < 1.0e-12 || nb < 1.0e-12) { return 0.0; }
    return dot / (sqrt(na) * sqrt(nb));
}

# ---------------------------------------------------------------------------
# Multinomial naive Bayes text classifier (Laplace smoothing).
# model = [classes, priorsLog, condLog(nClasses x nWords), vocab]
# ---------------------------------------------------------------------------

fn nb_fit(docsTokens, labels) {
    int c = 0; int cidx = 0; int classes = 0; int cond = 0; int counts = 0; int d = 0; int docCount = 0; int k = 0; int nDocs = 0; int nWords = 0; int nc = 0; int out = 0; int priors = 0; int row = 0; int totals = 0; int v = 0; int vocab = 0;
    vocab = build_vocab(docsTokens);
    nWords = len(vocab[0]);
    classes = [];
    cidx = dictx.dnew();
    for lb in labels {
        if (dictx.dhas(cidx, lb) == false) {
            cidx = dictx.dset(cidx, lb, len(classes));
            classes[len(classes)] = lb;
        }
    }
    nc = len(classes);
    counts = [];
    totals = arrayx.zerosf(nc);
    docCount = arrayx.zerosf(nc);
    for (c = 0; c < nc; ++c) { counts[c] = arrayx.zerosf(nWords); }
    for (d = 0; d < len(docsTokens); ++d) {
        c = dictx.dget(cidx, labels[d], 0);
        docCount[c] = docCount[c] + 1.0;
        v = bow_vector(docsTokens[d], vocab);
        row = counts[c];
        for (k = 0; k < nWords; ++k) {
            row[k] = row[k] + v[k];
            totals[c] = totals[c] + v[k];
        }
        counts[c] = row;
    }
    priors = [];
    cond = [];
    nDocs = len(docsTokens);
    for (c = 0; c < nc; ++c) {
        priors[c] = ln(docCount[c] / nDocs);
        row = counts[c];
        out = [];
        for (k = 0; k < nWords; ++k) {
            out[k] = ln((row[k] + 1.0) / (totals[c] + nWords));
        }
        cond[c] = out;
    }
    return [classes, priors, cond, vocab];
}

fn nb_predict(model, tokens) {
    int best = 0; int bestScore = 0; int c = 0; int classes = 0; int cond = 0; int k = 0; int priors = 0; int row = 0; int s = 0; int v = 0; int vocab = 0;
    classes = model[0];
    priors = model[1];
    cond = model[2];
    vocab = model[3];
    v = bow_vector(tokens, vocab);
    best = 0;
    bestScore = -1.0e30;
    for (c = 0; c < len(classes); ++c) {
        s = priors[c];
        row = cond[c];
        for (k = 0; k < len(v); ++k) {
            if (v[k] > 0.0) { s = s + v[k] * row[k]; }
        }
        if (s > bestScore) { bestScore = s; best = c; }
    }
    return classes[best];
}

# ---------------------------------------------------------------------------
# Lexicon sentiment with simple negation flipping.
# Returns score in [-1, 1].
# ---------------------------------------------------------------------------

fn sentiment(text) {
    int flip = 0; int hits = 0; int i = 0; int neg = 0; int negW = 0; int negations = 0; int ng = 0; int pos = 0; int posW = 0; int score = 0; int toks = 0; int v = 0; int w = 0;
    posW = ["good", "great", "excellent", "amazing", "wonderful", "love",
            "loved", "best", "fantastic", "happy", "delightful", "brilliant",
            "enjoy", "enjoyed", "beautiful", "perfect", "awesome", "nice",
            "superb", "impressive", "pleasant", "favorite", "win", "success"];
    negW = ["bad", "terrible", "awful", "horrible", "hate", "hated", "worst",
            "poor", "disappointing", "sad", "boring", "dreadful", "annoying",
            "broken", "fail", "failed", "failure", "ugly", "mediocre",
            "unpleasant", "regret", "waste", "lose", "loss"];
    negations = ["not", "never", "no", "isn't", "wasn't", "don't", "doesn't",
                 "didn't", "can't", "couldn't", "won't", "wouldn't", "hardly"];
    pos = dictx.dnew();
    neg = dictx.dnew();
    ng = dictx.dnew();
    for w in posW { pos = dictx.dset(pos, w, 1); }
    for w in negW { neg = dictx.dset(neg, w, 1); }
    for w in negations { ng = dictx.dset(ng, w, 1); }
    toks = tokenize(text);
    score = 0.0;
    hits = 0;
    flip = 1.0;
    for (i = 0; i < len(toks); ++i) {
        w = toks[i];
        if (dictx.dhas(ng, w)) { flip = 0.0 - flip; continue; }
        v = 0.0;
        if (dictx.dhas(pos, w)) { v = 1.0; }
        elif (dictx.dhas(neg, w)) { v = -1.0; }
        if (v != 0.0) {
            score = score + v * flip;
            hits = hits + 1;
            flip = 1.0;                  # negation applies to the next hit
        }
    }
    if (hits == 0) { return 0.0; }
    return mathx.clampf(score / hits, -1.0, 1.0);
}

# ---------------------------------------------------------------------------
# Order-k word-level Markov chain.
# model = [k, transitionsDict(context -> nextWordsArray), startsArray]
# ---------------------------------------------------------------------------

fn markov_fit(text, k) {
    int i = 0; int j = 0; int key = 0; int lst = 0; int n = 0; int nxt = 0; int starts = 0; int toks = 0; int trans = 0;
    if (k == nil) { k = 2; }
    toks = tokenize(text);
    trans = dictx.dnew();
    starts = [];
    n = len(toks);
    for (i = 0; i + k < n; ++i) {
        key = toks[i];
        for (j = 1; j < k; ++j) { key = key + " " + toks[i + j]; }
        nxt = toks[i + k];
        lst = dictx.dget(trans, key, nil);
        if (lst == nil) { lst = []; }
        lst[len(lst)] = nxt;
        trans = dictx.dset(trans, key, lst);
        if (i == 0 || toks[i - 1] == ".") { starts[len(starts)] = key; }
        if (len(starts) == 0) { starts[0] = key; }
    }
    return [k, trans, starts];
}

fn markov_generate(model, nWords, seed) {
    int ctx = 0; int j = 0; int k = 0; int lst = 0; int nxt = 0; int outWords = 0; int parts = 0; int r = 0; int starts = 0; int trans = 0;
    k = model[0];
    trans = model[1];
    starts = model[2];
    if (len(starts) == 0) { return ""; }
    r = prng.randint(seed, 0, len(starts) - 1);
    ctx = starts[r[0]];
    seed = r[1];
    outWords = strx.split(ctx, " ");
    while (len(outWords) < nWords) {
        lst = dictx.dget(trans, ctx, nil);
        if (lst == nil || len(lst) == 0) { break; }
        r = prng.randint(seed, 0, len(lst) - 1);
        seed = r[1];
        nxt = lst[r[0]];
        outWords[len(outWords)] = nxt;
        # slide the context window
        parts = strx.split(ctx, " ");
        ctx = "";
        for (j = 1; j < k; ++j) {
            if (j > 1) { ctx = ctx + " "; }
            ctx = ctx + parts[j];
        }
        if (k > 1) { ctx = ctx + " "; }
        ctx = ctx + nxt;
    }
    return strx.join(outWords, " ");
}

# ---------------------------------------------------------------------------
# Extractive summarizer: score sentences by mean TF weight of their
# non-stopword stems; return the top-n in original order.
# ---------------------------------------------------------------------------

fn summarize(text, nSentences) {
    int docs = 0; int i = 0; int keep = 0; int order = 0; int out = 0; int s = 0; int scores = 0; int sents = 0; int t = 0; int tf = 0; int toks = 0;
    if (nSentences == nil) { nSentences = 2; }
    sents = sentences(text);
    if (len(sents) <= nSentences) { return strx.join(sents, " "); }
    # global term frequency over stems
    tf = dictx.dnew();
    docs = [];
    for (i = 0; i < len(sents); ++i) {
        toks = stem_tokens(remove_stopwords(tokenize(sents[i])));
        docs[i] = toks;
        for w in toks { tf = dictx.dset(tf, w, dictx.dget(tf, w, 0) + 1); }
    }
    scores = [];
    for (i = 0; i < len(sents); ++i) {
        toks = docs[i];
        s = 0.0;
        for w in toks { s = s + dictx.dget(tf, w, 0); }
        if (len(toks) > 0) { s = s / len(toks); }
        scores[i] = s;
    }
    order = arrayx.argsort(scores);
    keep = dictx.dnew();
    for (t = 0; t < nSentences; ++t) {
        keep = dictx.dset(keep, string(order[len(order) - 1 - t]), 1);
    }
    out = [];
    for (i = 0; i < len(sents); ++i) {
        if (dictx.dhas(keep, string(i))) { out[len(out)] = sents[i]; }
    }
    return strx.join(out, " ");
}

# ---------------------------------------------------------------------------
# Levenshtein edit distance (two rolling rows).
# ---------------------------------------------------------------------------

fn levenshtein(a, b) {
    int cost = 0; int cur = 0; int d = 0; int i = 0; int ins = 0; int j = 0; int na = 0; int nb = 0; int prev = 0; int sub = 0;
    na = len(a);
    nb = len(b);
    if (na == 0) { return nb; }
    if (nb == 0) { return na; }
    prev = [];
    for (j = 0; j <= nb; ++j) { prev[j] = j; }
    for (i = 1; i <= na; ++i) {
        cur = [];
        cur[0] = i;
        for (j = 1; j <= nb; ++j) {
            cost = 1;
            if (a[i - 1] == b[j - 1]) { cost = 0; }
            d = prev[j] + 1;
            ins = cur[j - 1] + 1;
            if (ins < d) { d = ins; }
            sub = prev[j - 1] + cost;
            if (sub < d) { d = sub; }
            cur[j] = d;
        }
        prev = cur;
    }
    return prev[nb];
}
