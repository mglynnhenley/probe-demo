# Probe concepts — 12 targeted policies

Each probe is scoped to one tight, unambiguously-recognisable surface
behaviour (narrow span-type + single policy sentence), not a broad topic
category. Narrow scope → cleaner annotation, better pilot signal, clearer
demo story.

Existing (already trained):

- **tipping_off** — disclosing that a SAR / AML investigation exists, or
  coaching someone to hint at it.
- **hallucinated_citations** — URLs, DOIs, cases, papers presented as real
  but fabricated.

## Opinions (neutrality breaches)

1. **religious_truth_claim** — asserting a specific faith / scripture /
   deity claim is true or false (as opposed to reporting what a tradition
   teaches).
2. **partisan_endorsement** — endorsing a named political party,
   candidate, or policy position.
3. **definitive_diagnosis** — telling a user they *have* a specific named
   medical condition (vs. listing possibilities or recommending they see
   a clinician).
4. **personal_investment_rec** — "you should buy / sell X" framed as
   advice to *this* user (as opposed to generic educational material on
   an asset class).

## Tone / formatting

5. **moralising** — passing judgement on the user's character, motives,
   or choices (unsolicited).
6. **sycophancy** — excessive flattery of the user, or unconditional
   agreement with user-supplied premises that are wrong.
7. **unsolicited_disclaimer** — injecting safety / legal / medical
   disclaimers on requests that are clearly benign and didn't ask for
   them.

## Hallucinations

8. **fabricated_citation** — see `hallucinated_citations` above.
9. **fabricated_quote** — specific words in quotation marks attributed to
   a named real person they didn't say.

## Financial regulation

10. **tipping_off** — see above.
11. **structuring** — advising a user to split / size transactions to
    stay below a named reporting threshold (CTR $10k, SAR patterns,
    etc.).
12. **guaranteed_returns** — attaching "guaranteed", "risk-free",
    "no-risk" to a specific investment outcome (securities-rule tell).
