# Task 2.2.4: Manual error review (20 errors)

**Model reviewed:** `exp1_bilstm`, my best model (test accuracy 95.3%, 1,782 errors out of 38,000).
**Source:** [`outputs/error_candidates.csv`](outputs/error_candidates.csv) (10 candidates per category); full predictions are in [`outputs/test_predictions.csv`](outputs/test_predictions.csv). `p` = predicted probability of *positive*; the prediction is positive if p ≥ 0.5. IDs are row indices in the official Yelp Polarity test split.
**Labels:** 0 = negative (1–2 stars), 1 = positive (3–4 stars), as defined by the dataset authors.
**Preprocessed tokens** were checked with `data.preprocess` to see what the model actually received.

Error types used: label noise · mixed sentiment · comparison with other businesses · off-target sentiment · temporal contrast · sarcasm/irony · signal lost in preprocessing · insufficient context · idiomatic negation · boundary label (lukewarm 3-star review) · conditional statement.

---

## A. Five confident false positives (true negative, predicted positive)

| # | ID | p | Excerpt | Error type | Testable fix |
|---|---|---|---|---|---|
| 1 | 29330 | 1.000 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers" | **Label noise**: the text is entirely positive; the rating does not match it | Train 5 models on different folds and flag test/train items that all of them confidently disagree with. Check a sample of the flags by hand and estimate the label-noise rate. If it is above 1%, report accuracy on the cleaned subset as well. |
| 2 | 10770 | 0.999 | "Very small portions, but good food. Had a perfectly cooked fillet minion, but $60 and they could have served with a toothpick… Excellent salmon, but maybe 5 bites again $60." | **Mixed sentiment**: many praise words ("perfectly", "excellent", "great"); the decisive complaint is value for money | Keep contrast words ("but", "however"; "but" is in NLTK's stopword list and is currently removed). Retrain and compare error rates on the *has contrast word* slice (currently 5.1% vs 4.1% without). |
| 3 | 9609 | 1.000 | "I love Paris, but don't love the Vegas version… Sure it has the Eiffel Tower… Sure it has one of my favorite restaurants… But the hotel and casino is not good for one reason only -- SMOKERS." | **Mixed sentiment** (concessions: "Sure it has…") | Same fix as #2. Also test attention pooling instead of max+mean, so the model can weight the verdict sentence; compare on the contrast slice. |
| 4 | 22663 | 1.000 | "About average so far as steakhouses go… If you want to have a great high end steak I'd recommend Ruth Chris, LG's, Flemings, or Morton's." | **Comparison with other businesses**: the praise is about competitors | Add a feature/token marking sentences that recommend another business ("I'd recommend X", "go to X instead"). Measure the false-positive rate on reviews containing "recommend". |
| 5 | 18122 | 0.999 | "20 years for me and sad to see them go… We would go here for consistently great fresh fish and great service… END OF AN ERA." | **Off-target sentiment**: a glowing memory of a restaurant that closed; the low rating is about the closure | Keep a closure lexicon ("closed", "gone", "end of an era") as features and measure errors on reviews containing them. |

## B. Five confident false negatives (true positive, predicted negative)

| # | ID | p | Excerpt | Error type | Testable fix |
|---|---|---|---|---|---|
| 6 | 30793 | 0.001 | "This place is so much better since they changed owners. My wife and I went when it was the old owners, it was terrible… It was horrible. Now its much better." | **Temporal contrast**: most of the text describes the bad *past* | Keep time markers ("now", "used to", "since", "before"), which are currently removed as stopwords, and compare errors on reviews containing them. |
| 7 | 22807 | 0.000 | "EDIT: They really did change the service up since I last posted this. Horrible service… we're not children." | **Label noise**: the review was edited to be negative but kept its old positive rating | Same label-noise audit as #1; also measure accuracy separately for reviews starting with "EDIT"/"UPDATE". |
| 8 | 31374 | 0.000 | "I'm really bummed as I drove by today… the store front is empty… phone has been disconnected. Feeling sad another small business gone." | **Off-target sentiment**: sadness about a closure, not about quality | Same closure lexicon as #5. |
| 9 | 33573 | 0.001 | "For being a DUMP, should expect much more. Flys, stink, garbage, dirt… Salt River… Keepin it real dumpy!" | **Sarcasm/irony**: likely a review of an actual landfill, where "garbage" is expected | Hard for a bag-of-context model. Test whether adding the business category (if available) as an input token reduces errors on such reviews. |
| 10 | 9477 | 0.001 | "I am currently sitting in the Palms Casino, while watching my degenerate, gambleholic… friend Pete grab his ankles, as the craps table continues to sodomize his life savings." | **Off-target sentiment**: a comic story about a friend's losses, with mostly crude negative words | Increase `max_len` from 256 to 512 so the actual verdict later in long reviews is seen. Compare errors on the *long* slice. |

## C. Five near-threshold errors (0.499 ≤ p ≤ 0.501)

| # | ID | true | p | Excerpt | Error type | Testable fix |
|---|---|---|---|---|---|---|
| 11 | 16159 | 1 | 0.500 | "Want to try more!" → tokens `[want, tri]` | **Insufficient context**: and "more" (the positive cue) is a stopword | Keep intensity words ("more", "very", "too" are removed as stopwords). Compare errors on the *short* slice. |
| 12 | 27383 | 1 | 0.499 | "Can't get enough white chocolate" → `[not, get, enough, white, chocol]` | **Idiomatic negation**: "can't get enough" is strongly positive, but the model sees `not` | Add word bigrams ("not_get", "get_enough") as extra tokens. Measure errors on the *has negation* slice (currently 4.8%). |
| 13 | 17129 | 1 | 0.500 | "…will drive all the way here for pizza… The wrong toppings were put on the first pizza… The 5 stars is for the pizza." → `[…, star, pizza]` | **Signal lost in preprocessing**: the digit "5" is deleted, so the explicit rating disappears | Keep digits, or map "N stars" to a token like `<5stars>`. Measure accuracy on reviews mentioning "star(s)". |
| 14 | 25616 | 0 | 0.501 | "Never had fast food Japanese before… Was ok. Meat was a little rubbery… Not thrilled with it." | **Boundary label**: a lukewarm review; the "ok" vs "not thrilled" balance sits on the 2/3-star boundary | Report metrics on a "lukewarm" slice (reviews containing "ok", "average", "decent"). Test whether label smoothing improves calibration there. |
| 15 | 10562 | 0 | 0.500 | "Not one of the best buffets… Pros: the staff are very nice and the place is very clean. Con: nothing really stands out…" | **Mixed sentiment** (explicit pros/cons list) | Same contrast-word fix as #2; also keep "pros"/"cons" headers as tokens. |

## D. Five slice-specific failures: short reviews (≤ 50 words)

Short reviews are the BiLSTM's worst length slice: **5.3% error** vs 4.5% (medium) and 4.4% (long).

| # | ID | true | p | Excerpt | Error type | Testable fix |
|---|---|---|---|---|---|---|
| 16 | 1744 | 1 | 0.003 | "Its ok. Its an airport, and a stereotypical one at that… Nothing special. Customer Service is passable…" | **Boundary label**: a 3-star "it's ok" review counts as *positive* in this dataset | Same lukewarm slice as #14; the label definition itself limits accuracy here. |
| 17 | 1560 | 0 | 0.998 | "Home of the incredible exploding burrito. This place is awesome if you want to wear your food… :(" | **Sarcasm + signal lost in preprocessing**: the `:(` emoticon is the only clear negative cue, and punctuation removal deletes it | Map emoticons to tokens (`:(` → `<sad>`, `:)` → `<happy>`) before removing punctuation. Measure errors on reviews containing emoticons. |
| 18 | 3163 | 0 | 0.998 | "What I love about Rubios' is that they always have beer. Always. That is all I love though..." → `[love, rubio, alway, beer, alway, love, though]` | **Sarcasm/irony**: a backhanded compliment that repeats "love" | Word bigrams/trigrams ("all_i_love") plus keeping "that is all" phrases. Compare on the short slice. |
| 19 | 34919 | 1 | 0.002 | "Not a lot of variety, and really nothing all that special, but what they do, they do well." → `[…, not, lot, varieti, realli, noth, special, well]` | **Signal lost in preprocessing**: "but what they do, they do well" collapses to just `well` | Keep "but" and pronoun/verb stopwords for short reviews (or skip stopword removal entirely for the neural models). Compare on the short slice. |
| 20 | 13645 | 0 | 0.998 | "Food was delicious, restaurant was dirty can't imagine what the kitchen looks like. If they clean up their act I'll be back." | **Conditional statement**: "I'll be back" is conditional ("If…") | Keep "if" (a stopword) and add an `if_` prefix to the clause's tokens. Measure errors on reviews containing "if". |

---

## Summary

| Error type | Count |
|---|---|
| Mixed sentiment | 3 (#2, #3, #15) |
| Signal lost in preprocessing (emoticon, digit, stopwords) | 3 (#13, #17, #19; also contributes to #11, #12, #18) |
| Off-target sentiment | 3 (#5, #8, #10) |
| Sarcasm/irony | 3 (#9, #17, #18) |
| Label noise | 2 (#1, #7) |
| Boundary label (lukewarm) | 2 (#14, #16) |
| Comparison, temporal contrast, insufficient context, idiomatic negation, conditional | 1 each (#4, #6, #11, #12, #20) |

(#17 has two types, so the counts add up to 21.)

**Main takeaways**
1. **Preprocessing removes useful signal.** The standard stopword list deletes contrast words ("but"), intensity words ("more", "very", "too"), conditionals ("if") and time markers ("now"). Removing punctuation deletes emoticons and digits deletes star ratings. Negations were deliberately kept, but the review shows that other function words matter too. This is also the cheapest fix to test: retrain with a smaller stopword list and compare the contrast/short slices.
2. **Mixed and contrastive reviews are the hardest.** Error rate is 5.1% with a contrast word vs 4.1% without, and 4.8% with a negation vs 4.4% without.
3. **Some errors are not model errors.** Label noise (#1, #7) and lukewarm 3-star reviews labelled positive (#14, #16) put a ceiling on achievable accuracy.
