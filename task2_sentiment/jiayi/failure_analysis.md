# Task 2 Manual Error Analysis

## Evaluation setup
This analysis used experiment 1 mlp model. The decision threshold is 0.5000 with >= 0.5000 is positive. 
The input length used for training is 185 processed tokens. Reviews longer than this are truncated before inference, so longer review may be unavailable to the model.
Positive-class probability is denoted as P(positive).

## 1. Confident False Positives

### Example 11429 — Confident False Positive

- Ground truth: Negative
- Prediction: Positive
- P(positive): 0.999998
- Threshold: 0.5000
- Error type: `fail to recognize irony`
- Review: “Yay. Cafeteria food. Yay.”
- Observation: The model appears to interpret the repeated positive word “Yay” literally and misses the irony part of the review.
- Testable fix: Retain punctuation/repetition information during preprocessing.

### Example 4742 — Confident False Positive

- Ground truth: Negative
- Prediction: Positive
- P(positive): 0.999997
- Threshold: 0.5000
- Error type: `fail to recognize comparative reversal`
- Review: “I have found them to be less than accomodating or helpful.”
- Observation: Positive words such as “accomodating” and “helpful” dominate the prediction, while “less than” reverses their meaning.
- Testable fix: Preserve comparative terms during preprocessing then compare F1 on reviews containing phrases such as “less than”.

### Example 20681 — Confident False Positive

- Ground truth: Negative
- Prediction: Positive
- P(positive): 0.999967
- Threshold: 0.5000
- Error type: `insufficient context, label ambiguity`
- Review: “Large variety”
- Observation: The review is extremely short and sounds neutral or even positive by itself, so there is little context to explain the negative label.
- Testable fix: Excluding verified ambiguous-label cases.

### Example 17640 — Confident False Positive

- Ground truth: Negative
- Prediction: Positive
- P(positive): 0.999947
- Threshold: 0.5000
- Error type: `fail to recognize comparative reversal`
- Review: “of all the buffets I had in Vegas this was my least favorite!”
- Observation: “Favorite” and the exclamation mark provide strong positive cues, but “least” reverses the phrase to negative sentiment.
- Testable fix: Preserve comparative words such as “least” then evaluate reviews containing comparative sentiment phrases.

### Example 3163 — Confident False Positive

- Ground truth: Negative
- Prediction: Positive
- P(positive): 0.999933
- Threshold: 0.5000
- Error type: `mixed sentiment contrast`
- Review: “What I love about Rubios' is that they always have beer. Always. That is all I love though...”
- Observation: The review uses “love” twice, but the actual meaning is that beer is the only thing the reviewer likes. The model appears to overweight the positive words and miss the contrast.
- Testable fix: Preserve contrast terms such as “though” and compare with an order-sensitive model on reviews containing mixed positive and negative sentiment.

## 2. Confident False Negatives

### Example 1131 — Confident False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.000002
- Threshold: 0.5000
- Error type: `fail to recognize idiom expression`
- Review: “The margaritas wont do ya dirty!”
- Observation: “Dirty” is usually negative, but the idiom “won't do ya dirty” means the margaritas will not disappoint. The model appears to rely on the individual negative word instead of the phrase meaning.
- Testable fix: Add bigram features or more informal Yelp expressions during training.

### Example 22807 — Confident False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.000260
- Threshold: 0.5000
- Error type: `event ordering`
- Review excerpt: “EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza...”
- Observation: The positive update appears at the beginning, but the older review body contains strongly negative wording such as “Horrible service.” The model may not represent the update-versus-old-review relationship well.
- Testable fix: Preserve edit: markers.

### Example 22707 — Confident False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.000354
- Threshold: 0.5000
- Error type: `mixed sentiment contrast`
- Review: “Its a nice buffet looking over the Aria pool, but variety was just ok.... Meh..”
- Observation: The review starts positively with “nice buffet” but then shifts through “but”, “just ok”, and “Meh”. The text itself contains mixed cues, and the model strongly favors the negative side.
- Testable fix: Preserve contrast words such as “but” and evaluate performance on a slice containing both positive and negative sentiment cues.

### Example 30793 — Confident False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.000419
- Threshold: 0.5000
- Error type: `event ordering`
- Review excerpt: “This place is so much better since they changed owners... it was terrible... Now its much better... I have nothing but positive things to now say...”
- Observation: The review contrasts a bad experience under the old owners with a much better current experience. The current sentiment is positive, but the model appears heavily influenced by the earlier negative description.
- Testable fix: Compare the current mean-pooling MLP with an order-sensitive sequence model and measure performance on reviews containing old/new or before/after sentiment shifts.

### Example 19857 — Confident False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.000595
- Threshold: 0.5000
- Error type: `contextual phrase ambiguity`
- Review: “Um... left speechless. I'll head back to Montreal for the sole purpose of returning here. I can't really write much else.”
- Observation: “Speechless” can be positive or negative depending on context. The later sentence about returning makes the meaning positive, but the model may interpret “speechless” negatively in isolation.
- Testable fix: Add phrase-level or n-gram features.

## 3. Near-Threshold Errors

### Example 29338 — Near-Threshold Error

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.499660
- Threshold: 0.5000
- Distance from threshold: 0.000340
- Error type: `context dependent sentiment`
- Review excerpt: “Vulgar, racist, and hilarious... But if you're open-minded, be sure to stop by.”
- Observation: Words such as “vulgar” and “racist” normally sound negative, but this reviewer enjoys that style and ultimately recommends the show. The model appears conflicted by the usual polarity of the individual words.
- Testable fix: Add phrase-level or n-gram features and evaluate on reviews where normally negative words are used in an approving context.

### Example 18707 — Near-Threshold Error

- Ground truth: Negative
- Prediction: Positive
- P(positive): 0.500531
- Threshold: 0.5000
- Distance from threshold: 0.000531
- Error type: `late sentiment shift`
- Review excerpt: “The fresh fruit was good. I wouldn't pay for the meal and probably won't be back.”
- Observation: Much of the review is factual or mildly positive, including “fresh fruit was good,” while the final sentences reveal the overall negative judgment.
- Testable fix: Compare with an order-sensitive model or a pooling scheme that gives more weight to later sentences, then evaluate reviews whose sentiment changes near the end.

### Example 7160 — Near-Threshold Error

- Ground truth: Negative
- Prediction: Positive
- P(positive): 0.500608
- Threshold: 0.5000
- Distance from threshold: 0.000608
- Error type: `negative cue dilution`
- Review excerpt: “The line we were all in had at least 400 people in it standing outside... in the chilly air... Not good!”
- Observation: Most of the review describes the situation, while the explicit negative judgment is brief. The probability is already almost exactly at the threshold, suggesting the negative cue is diluted by the surrounding text.
- Testable fix: Preserve negation and sentence-level sentiment cues, then compare performance on reviews with short negative conclusions.

### Example 5900 — Near-Threshold Error

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.499205
- Threshold: 0.5000
- Distance from threshold: 0.000795
- Error type: `mixed sentiment balance`
- Review excerpt: “it's good but it ain't THAT good... above average... good selection... really tasty soups... excellent quiche.”
- Observation: The review opens with a hesitant or negative-sounding qualification, but later contains several positive evaluations. The model is almost evenly balanced between the conflicting cues.
- Testable fix: Compare with an order-sensitive model and evaluate a slice containing reviews with both positive and negative statements.

### Example 1734 — Near-Threshold Error

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.499109
- Threshold: 0.5000
- Distance from threshold: 0.000891
- Error type: `sentiment shift contrast`
- Review excerpt: “Crazy busy airport with long wait times at security... That said however, this is a pretty decent airport... Go PHX!”
- Observation: The review begins negatively but explicitly reverses direction with “That said however” and ends positively. The model appears slightly more influenced by the negative opening.
- Testable fix: Preserve contrast markers such as “however”

## 4. Slice-Specific Long-Review Failures

### Example 30958 — Slice-Specific Long Failure / False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.002222
- Threshold: 0.5000
- Raw review length: 397 words
- Slice: Long
- Error type: `long review truncation`
- Review excerpt: Early: “Cox sucks... terrible business...” Later: “why do I give this store 5 stars? The Manager, Mario... That's customer service... my opinion of Cox has changed.”
- Observation: The review contains a large amount of strongly negative material before the positive resolution. Because inference uses only the first 185 processed tokens, important positive evidence near the end may be truncated.
- Testable fix: Increase `MAX_LEN` from 185 to 256, retrain the same MLP with all other settings fixed, and compare Macro-F1 and error rate on the long-review slice.

### Example 20061 — Slice-Specific Long Failure / False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.003109
- Threshold: 0.5000
- Raw review length: 597 words
- Slice: Long
- Error type: `long review truncation`
- Review excerpt: Middle: “the place stinks... the ground is wet and a bit slippery... the bins were full...” Ending: “we got to rid ourselves of the massive piles... in one fell swoop. Can't beat that!”
- Observation: The review describes many unpleasant details about a waste facility, but its overall judgment is positive and the strongest positive conclusion occurs late in the review. The model may not receive that ending after truncation.
- Testable fix: Increase `MAX_LEN` from 185 to 256, retrain the same MLP, and compare long-slice Macro-F1/error rate with the original run.

### Example 21813 — Slice-Specific Long Failure / False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.003778
- Threshold: 0.5000
- Raw review length: 553 words
- Slice: Long
- Error type: `long review truncation mixed sentiment`
- Review excerpt: Earlier: room delays, trash, dirty utensils, and unanswered calls. Later: “The hotel is a nice hotel...” and “The entire property is nice, which I would stay there again...”
- Observation: The review contains substantial negative experience details, while the positive overall judgment is expressed later. Truncation may remove or weaken the positive conclusion.
- Testable fix: Increase `MAX_LEN` to 256 while holding the model and hyperparameters fixed, then compare the long-slice metrics and specifically re-check this example.

### Example 29494 — Slice-Specific Long Failure / False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.004915
- Threshold: 0.5000
- Raw review length: 322 words
- Slice: Long
- Error type: `review update truncation`
- Review excerpt: Original review: “Billing practices are at best negligent and at worst fraudulent.” Update: “this business went out of its way to accomodate and help me... I am thus updating my review to 4 stars...”
- Observation: The first part is an old strongly negative review, while the later update changes the final evaluation to positive. Since the model reads only the first 185 processed tokens, some or all of the corrective update may be unavailable.
- Testable fix: Increase `MAX_LEN` from 185 to 256 and compare performance on reviews containing “UPDATED”, “EDIT”, or other explicit review revisions.

### Example 12280 — Slice-Specific Long Failure / False Negative

- Ground truth: Positive
- Prediction: Negative
- P(positive): 0.005034
- Threshold: 0.5000
- Raw review length: 395 words
- Slice: Long
- Error type: `target confusion`
- Review excerpt: Venue assessment: “Not the best accoustics, but still a fairly nice venue... It's not the venue's fault...” Most later negative language describes the behavior of another couple at the concert.
- Observation: Much of the negative language targets other concertgoers rather than the venue itself. The model appears to aggregate negative words without reliably separating the sentiment target; the long input may further dilute the relevant venue assessment.
- Testable fix: Compare the current mean-pooling MLP with a sentence-level or attention-based aggregation method.
