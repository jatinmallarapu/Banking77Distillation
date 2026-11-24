# Initial CPU error analysis

Measured on the fixed 3,080-query official test set. Source text is PolyAI Banking77, CC BY 4.0; see ATTRIBUTION.md. Examples are masked. These are descriptive results, not a justification for retuning against this test set.

## Five lowest-F1 intents

| Intent | F1 | 95% bootstrap interval | Support |
| --- | ---: | --- | ---: |
| balance_not_updated_after_bank_transfer | 0.697 | 0.596–0.782 | 40 |
| pending_transfer | 0.704 | 0.571–0.805 | 40 |
| card_not_working | 0.747 | 0.673–0.830 | 40 |
| wrong_exchange_rate_for_cash_withdrawal | 0.773 | 0.667–0.874 | 40 |
| transfer_not_received_by_recipient | 0.776 | 0.678–0.866 | 40 |

## Ten most frequent directional confusions

### card_acceptance → card_not_working (8)

- `test-00979`: I am traveling to Germany, Will I be able to use my card there?
- `test-00984`: Is it acceptable to use my card anywhere?
- `test-00988`: Where should my card work?

### why_verify_identity → verify_my_identity (6)

- `test-01161`: I'd rather not verify my identity.
- `test-01183`: Do I really need to verify my identity?
- `test-01187`: I won't verify my identity.

### virtual_card_not_working → getting_virtual_card (6)

- `test-02525`: Why is my virtual card is being declined?
- `test-02532`: Why was my virtual card declined when attempting to setup automatic billing?
- `test-02536`: I cannot make transactions with my virtual card.

### top_up_reverted → top_up_failed (5)

- `test-01003`: My top-up showed as complete, but it's no longer there! What's going on here?
- `test-01004`: Hi, i don't know what's going on i've just paid for my top up twice by accident can you help me get a refund please.
- `test-01006`: The app wouldn't accept my top up.

### unable_to_verify_identity → verify_my_identity (5)

- `test-01200`: I cannot verify my identity
- `test-01220`: I can't verify my identity.
- `test-01225`: What do i need to verify my id?

### pending_transfer → transfer_timing (5)

- `test-01857`: How long do transfer pend for?
- `test-01859`: When will the transfer go through?
- `test-01864`: When will the transfer be completed?

### fiat_currency_support → exchange_via_app (4)

- `test-00255`: How do I change currencies to euros?
- `test-00264`: Am I able to exchange currencies?
- `test-00268`: I want to make a currency exchange to EU.

### pin_blocked → get_physical_card (4)

- `test-00521`: Where can I view my PIN?
- `test-00522`: Will you reinstate my PIN?
- `test-00551`: Can I reactivate my PIN?

### contactless_not_working → card_not_working (4)

- `test-00561`: My contanctless has stopped working
- `test-00563`: I don't know what's wrong, my contactless stopped working. Tried it in a few different places today and it didn't work in any of them.
- `test-00583`: The NFC payment wouldn't work on the bus today. Help?

### pending_top_up → top_up_failed (4)

- `test-00641`: My card was topped this morning but I can't see the funds. Why didn't it complete?
- `test-00666`: OMG!  I'm trying to load my card and it wont top up!  I desperately need the money either on my card or in my bank, where is it?
- `test-00671`: It doesn't look like the top-up completed.

## Interpretation and next measurements

Related transfer-status and card-failure intents remain difficult. Lexical overlap is a plausible contributor, but these examples do not establish a causal diagnosis. The baseline is underconfident (ECE 0.2668); its threshold is validation-selected and should not be mistaken for a calibrated probability guarantee.

This initial analysis predates the GPU pilot. Both single-seed students now have complete test reports; see [the pilot findings](STUDENT_PILOT_FINDINGS.md). The three-seed comparison remains future work. Do not infer that augmentation fixes the CPU errors merely because a student was trained.
