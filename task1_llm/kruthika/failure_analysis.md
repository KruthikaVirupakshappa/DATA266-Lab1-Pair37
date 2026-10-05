# Task 1.4: Sequence model failure analysis

**Model:** `gpt_L6_H6_D384`, a character-level GPT (6 layers, 6 heads, 384-dim, 10.8M parameters, 256-character context).
**Checkpoint:** `checkpoints/gpt_L6_H6_D384_best.pt` (step 27,380, validation loss 0.536).
**Samples:** all from [`outputs/gpt_L6_H6_D384/samples.txt`](outputs/gpt_L6_H6_D384/samples.txt). There are 5 prompts, each with 4 temperature samples (T = 0.8) and 1 greedy sample, 500 new characters each.

Overall the text is fluent and grammatical. The failures are about *meaning over longer spans*, not spelling or grammar.

---

## Case 1: Repetition loop (greedy decoding)

**Prompt:** `Tom and his mom went to the park.` (greedy, temperature 0)

> Tom did not listen. He ran to the slide. He slid down fast. He felt the wind in his hair. He laughed.
>
> "Sam, you are silly. You are not silly. You are silly. You are not silly. You are silly. You are not silly. You are silly. You are not silly. You are

**Failure type:** Repetition.

**Observation:**
- **What happens.** The story is fine until the dialogue, then it falls into a two-sentence loop that repeats until the 500-character limit.
- **Why greedy decoding loops.** Greedy decoding always picks the single most likely next character. After "You are silly. You are not silly." the most likely continuation is the same pattern again, because repeated phrases are common in TinyStories. Each repetition makes the next one even more likely, so the model never escapes.
- **The metrics confirm it.** The repeated 4-gram rate is **8.3% for greedy** samples but only **0.19% for temperature samples**. Sampling adds randomness, which breaks these loops.
- **Possible fix:** use temperature or top-k sampling (as in our main samples), or add a repetition penalty.

---

## Case 2: Character inconsistency and a made-up character

**Prompt:** `In a big forest, there lived` (sampled, T = 0.8)

> In a big forest, there lived an old lady. She had a big red ball that kept her forever. The old lady always kicked the ball hard. The old lady was very happy.
>
> One day, a little boy named Tim came to visit. They were both sad because they could not play with the ball anymore. Tim asked the old lady, "Are you okay?" The old lady said, "No, I'm not okay. **I am a jolly boy.**" **Sue** was very sad.

**Failure type:** Loss of coherence: character/gender inconsistency and hallucination.

**Observation:**
- **The gender slip is not a memory problem.** The old lady calls herself "a jolly boy". The words "the old lady" appear only **78 characters** earlier, well inside the 256-character context window. The model can see her; it simply follows a strong local pattern. "I am a ..." is often followed by "boy" or "girl" in children's stories, and "boy" was also mentioned recently (Tim).
- **Tracking who is who is weak.** A character-level model learns spelling and grammar very well, but keeping track of who is who is much harder at this size.
- **"Sue" is a hallucinated character.** She is never introduced; the model inserts a common TinyStories name.
- **"They were both sad"** also contradicts the previous paragraph ("The old lady was very happy").
- **Possible fix:** a larger model, or more training, to learn longer-range entity tracking.

---

## Case 3: Forgetting who the character is (beyond the context window)

**Prompt:** `Once upon a time` (sampled, T = 0.8)

> Once upon a time, there was **a boy named Timmy**. He loved to play outside and run in the park. One morning, he saw a tree with a big trunk. [...] So, he started to climb the tree. He was having lots of fun until he saw a beautiful flower. He picked it up and started to play with it. He threw it in the air and **caught it with his long tail**. He was so happy when he touched it!

**Failure type:** Loss of coherence: contradiction with earlier context.

**Observation:**
- **The contradiction.** Timmy is introduced as a boy, but later catches a flower "with his long tail", as if he were an animal.
- **Why: the context limit.** "a boy named Timmy" is **377 characters** before "long tail", but the model only sees the last **256 characters** (`block_size`). When it writes "long tail", the information that Timmy is a boy has already dropped out of its view. It only sees "he", "tree", "climb" and "play", which in TinyStories often describe animals (monkeys, cats, squirrels).
- **Why the limit is tight.** At the character level, 256 characters is only about 50 words, so the model's memory is short in terms of story content.
- **Possible fix:** a longer context window (for example 512 characters), or word/subword tokens, so the same window covers more of the story.

---

## Summary

| # | Failure type | Decoding | Main cause |
|---|---|---|---|
| 1 | Repetition | greedy | always picking the most likely character reinforces loops |
| 2 | Character inconsistency + hallucination | T = 0.8 | strong local word patterns override who the character is |
| 3 | Contradiction with earlier context | T = 0.8 | key fact was outside the 256-character context window |
