# emx-scope-safe
# =============================================================================
# strx.em - string utilities for Emerald
#
# Emerald strings are immutable values; s[i] yields a char, s.get(i, j) an
# inclusive substring. All helpers here return new strings/arrays.
# =============================================================================

fn is_digit(c)  { return c >= '0' && c <= '9'; }
fn is_lower(c)  { return c >= 'a' && c <= 'z'; }
fn is_upper(c)  { return c >= 'A' && c <= 'Z'; }
fn is_alpha(c)  { return is_lower(c) || is_upper(c); }
fn is_alnum(c)  { return is_alpha(c) || is_digit(c); }
fn is_space(c)  { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
fn is_vowel(c)  {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'
        || c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
}

# Substring by start index and count (safe at the edges).
fn substr(s, start, count) {
    int last = 0; int n = 0;
    n = len(s);
    if (start < 0) { start = 0; }
    if (start >= n || count <= 0) { return ""; }
    last = start + count - 1;
    if (last > n - 1) { last = n - 1; }
    return s.get(start, last);
}

fn char_str(c) { return string(c); }

fn to_upper(s) {
    int c = 0; int i = 0; int out = 0;
    out = "";
    for (i = 0; i < len(s); ++i) {
        c = s[i];
        if (is_lower(c)) { out = out + string(char(int(c) - 32)); }
        else { out = out + string(c); }
    }
    return out;
}

fn to_lower(s) {
    int c = 0; int i = 0; int out = 0;
    out = "";
    for (i = 0; i < len(s); ++i) {
        c = s[i];
        if (is_upper(c)) { out = out + string(char(int(c) + 32)); }
        else { out = out + string(c); }
    }
    return out;
}

fn capitalize(s) {
    int head = 0;
    if (len(s) == 0) { return s; }
    head = to_upper(s.get(0, 0));
    if (len(s) == 1) { return head; }
    return head + s.get(1, len(s) - 1);
}

fn ltrim(s) {
    int i = 0; int n = 0;
    i = 0;
    n = len(s);
    while (i < n && is_space(s[i])) { i = i + 1; }
    if (i >= n) { return ""; }
    return s.get(i, n - 1);
}

fn rtrim(s) {
    int j = 0; int n = 0;
    n = len(s);
    j = n - 1;
    while (j >= 0 && is_space(s[j])) { j = j - 1; }
    if (j < 0) { return ""; }
    return s.get(0, j);
}

fn trim(s) { return ltrim(rtrim(s)); }

fn starts_with(s, prefix) {
    int lp = 0;
    lp = len(prefix);
    if (lp == 0) { return true; }
    if (len(s) < lp) { return false; }
    return s.get(0, lp - 1) == prefix;
}

fn ends_with(s, suffix) {
    int ls = 0; int n = 0;
    ls = len(suffix);
    if (ls == 0) { return true; }
    n = len(s);
    if (n < ls) { return false; }
    return s.get(n - ls, n - 1) == suffix;
}

# First index of `sub` at or after `from`; -1 if absent.
fn index_of(s, sub, fromIdx) {
    int i = 0; int j = 0; int last = 0; int m = 0; int n = 0; int ok = 0;
    if (fromIdx == nil) { fromIdx = 0; }
    n = len(s);
    m = len(sub);
    if (m == 0) { return fromIdx; }
    last = n - m;
    for (i = fromIdx; i <= last; ++i) {
        ok = true;
        for (j = 0; j < m; ++j) {
            if (!(s[i + j] == sub[j])) { ok = false; break; }
        }
        if (ok) { return i; }
    }
    return -1;
}

fn contains(s, sub) { return index_of(s, sub, 0) >= 0; }

fn count_of(s, sub) {
    int c = 0; int i = 0;
    c = 0;
    i = index_of(s, sub, 0);
    while (i >= 0) { c = c + 1; i = index_of(s, sub, i + len(sub)); }
    return c;
}

fn replace_all(s, fromS, toS) {
    int i = 0; int j = 0; int n = 0; int out = 0;
    if (len(fromS) == 0) { return s; }
    out = "";
    i = 0;
    n = len(s);
    while (i < n) {
        j = index_of(s, fromS, i);
        if (j < 0) {
            out = out + s.get(i, n - 1);
            return out;
        }
        if (j > i) { out = out + s.get(i, j - 1); }
        out = out + toS;
        i = j + len(fromS);
    }
    return out;
}

fn repeat(s, times) {
    int out = 0; int t = 0;
    out = "";
    for (t = 0; t < times; ++t) { out = out + s; }
    return out;
}

fn pad_left(s, width, padChar) {
    if (padChar == nil) { padChar = " "; }
    while (len(s) < width) { s = padChar + s; }
    return s;
}

fn pad_right(s, width, padChar) {
    if (padChar == nil) { padChar = " "; }
    while (len(s) < width) { s = s + padChar; }
    return s;
}

fn reverse(s) {
    int i = 0; int out = 0;
    out = "";
    for (i = len(s) - 1; i >= 0; --i) { out = out + string(s[i]); }
    return out;
}

# Split on a non-empty delimiter string. "a,,b" -> ["a", "", "b"].
fn split(s, delim) {
    int i = 0; int j = 0; int n = 0; int out = 0;
    out = [];
    n = len(s);
    if (len(delim) == 0) {
        for (i = 0; i < n; ++i) { out[i] = string(s[i]); }
        return out;
    }
    i = 0;
    while (i <= n) {
        j = index_of(s, delim, i);
        if (j < 0) {
            if (i <= n - 1) { out[len(out)] = s.get(i, n - 1); }
            else { out[len(out)] = ""; }
            return out;
        }
        if (j > i) { out[len(out)] = s.get(i, j - 1); }
        else { out[len(out)] = ""; }
        i = j + len(delim);
    }
    return out;
}

# Split on runs of whitespace; never yields empty tokens.
fn split_ws(s) {
    int i = 0; int j = 0; int n = 0; int out = 0;
    out = [];
    n = len(s);
    i = 0;
    while (i < n) {
        while (i < n && is_space(s[i])) { i = i + 1; }
        j = i;
        while (j < n && !is_space(s[j])) { j = j + 1; }
        if (j > i) { out[len(out)] = s.get(i, j - 1); }
        i = j;
    }
    return out;
}

fn split_lines(s) {
    s = replace_all(s, "\r\n", "\n");
    s = replace_all(s, "\r", "\n");
    return split(s, "\n");
}

fn join(arr, sep) {
    int i = 0; int out = 0;
    out = "";
    for (i = 0; i < len(arr); ++i) {
        if (i > 0) { out = out + sep; }
        out = out + string(arr[i]);
    }
    return out;
}

# ---------------------------------------------------------------------------
# Number parsing / formatting
# ---------------------------------------------------------------------------

# Parse an integer; returns nil if the string is not a valid integer.
fn parse_int(s) {
    int c = 0; int i = 0; int n = 0; int sgn = 0; int v = 0;
    s = trim(s);
    n = len(s);
    if (n == 0) { return nil; }
    i = 0;
    sgn = 1;
    if (s[0] == '-') { sgn = -1; i = 1; }
    elif (s[0] == '+') { i = 1; }
    if (i >= n) { return nil; }
    v = 0;
    while (i < n) {
        c = s[i];
        if (!is_digit(c)) { return nil; }
        v = v * 10 + (int(c) - 48);
        i = i + 1;
    }
    return sgn * v;
}

# Parse a float with optional sign, decimals, and exponent. nil on failure.
fn parse_float(s) {
    int esgn = 0; int ev = 0; int i = 0; int n = 0; int sawDigit = 0; int sawE = 0; int scale = 0; int sgn = 0; int v = 0;
    s = trim(s);
    n = len(s);
    if (n == 0) { return nil; }
    i = 0;
    sgn = 1.0;
    if (s[0] == '-') { sgn = -1.0; i = 1; }
    elif (s[0] == '+') { i = 1; }
    v = 0.0;
    sawDigit = false;
    while (i < n && is_digit(s[i])) {
        v = v * 10.0 + (int(s[i]) - 48);
        sawDigit = true;
        i = i + 1;
    }
    if (i < n && s[i] == '.') {
        i = i + 1;
        scale = 0.1;
        while (i < n && is_digit(s[i])) {
            v = v + (int(s[i]) - 48) * scale;
            scale = scale * 0.1;
            sawDigit = true;
            i = i + 1;
        }
    }
    if (!sawDigit) { return nil; }
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        i = i + 1;
        esgn = 1;
        if (i < n && s[i] == '-') { esgn = -1; i = i + 1; }
        elif (i < n && s[i] == '+') { i = i + 1; }
        ev = 0;
        sawE = false;
        while (i < n && is_digit(s[i])) {
            ev = ev * 10 + (int(s[i]) - 48);
            sawE = true;
            i = i + 1;
        }
        if (!sawE) { return nil; }
        v = v * (10.0 ** (esgn * ev));
    }
    if (i != n) { return nil; }
    return sgn * v;
}

fn is_number(s) { return parse_float(s) != nil; }

# Fixed-point formatting: fmt_f(3.14159, 2) -> "3.14". Handles negatives.
fn fmt_f(x, decimals) {
    int fp = 0; int fs = 0; int i = 0; int ip = 0; int m = 0; int neg = 0; int out = 0; int scaled = 0;
    if (decimals == nil) { decimals = 4; }
    neg = false;
    if (x < 0) { neg = true; x = 0.0 - x; }
    m = 1;
    for (i = 0; i < decimals; ++i) { m = m * 10; }
    scaled = floor(x * m + 0.5);
    ip = scaled // m;
    fp = scaled % m;
    out = string(ip);
    if (decimals > 0) {
        fs = string(fp);
        fs = pad_left(fs, decimals, "0");
        out = out + "." + fs;
    }
    if (neg && scaled != 0) { out = "-" + out; }
    return out;
}

# Join a numeric array as text with a fixed number of decimals.
fn fmt_arr(arr, decimals, sep) {
    int i = 0; int out = 0;
    if (sep == nil) { sep = " "; }
    out = "";
    for (i = 0; i < len(arr); ++i) {
        if (i > 0) { out = out + sep; }
        out = out + fmt_f(arr[i], decimals);
    }
    return out;
}

# Compact scientific-free float for file formats (trims trailing zeros).
fn fmt_g(x) {
    int j = 0; int s = 0;
    s = fmt_f(x, 6);
    # strip trailing zeros then a trailing dot
    j = len(s) - 1;
    while (j > 0 && s[j] == '0') { j = j - 1; }
    if (s[j] == '.') { j = j - 1; }
    return s.get(0, j);
}
