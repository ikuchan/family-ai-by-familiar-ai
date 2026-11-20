"""記-ad：記録の本文から事実を抜く頼みを組み、返りを読み、鍵の文にする（純関数・`core/fact_keys`）。"""

from __future__ import annotations

from familiar_agent.core import fact_keys as fk


def test_the_prompt_numbers_each_record_and_cuts_long_ones():
    p = fk.build_prompt(["町内会の夏祭りは8月3日", "あ" * 3000])
    assert "[0] 町内会の夏祭りは8月3日" in p
    assert "[1] " + "あ" * fk.MAX_CHARS in p
    assert "あ" * (fk.MAX_CHARS + 1) not in p


def test_newlines_in_a_record_do_not_break_the_numbering():
    p = fk.build_prompt(["一行目\n二行目"])
    assert "[0] 一行目 二行目" in p


def test_reads_facts_per_record():
    raw = '```json\n[{"i": 0, "facts": [{"s": "町内会", "r": "夏祭りの日", "a": "8月3日"}]}, {"i": 1, "facts": []}]\n```'
    got = fk.parse(raw, 2)
    assert got == [[fk.Fact("町内会", "夏祭りの日", "8月3日")], []]


def test_a_record_the_reply_skipped_stays_pending():
    """返りに番号の無い記録は None（印を付けず、次に抜き直す）。事実 0 個とは扱わない。"""
    got = fk.parse('[{"i": 1, "facts": [{"s": "x", "r": "y", "a": "z"}]}]', 2)
    assert got == [None, [fk.Fact("x", "y", "z")]]


def test_facts_without_a_relation_or_an_answer_are_dropped_and_at_most_eight_are_kept():
    many = ",".join('{"s": "s", "r": "r%d", "a": "a"}' % k for k in range(12))
    raw = '[{"i": 0, "facts": [{"s": "s", "r": "", "a": "a"}, {"s": "s", "r": "r"}, ' + many + "]}]"
    got = fk.parse(raw, 1)
    assert len(got[0]) == fk.MAX_FACTS == 8
    assert got[0][0] == fk.Fact("s", "r0", "a")


def test_an_unreadable_reply_is_none_so_the_records_stay_pending():
    assert fk.parse("すみません、できません", 2) is None
    assert fk.parse("", 2) is None
    assert fk.parse('{"i": 0}', 1) is None


def test_numbers_outside_the_batch_are_ignored():
    got = fk.parse(
        '[{"i": 5, "facts": [{"s": "x", "r": "y", "a": "z"}]}, {"i": "0", "facts": "x"}]', 1
    )
    assert got is None  # どの項も番号が当たらない


def test_one_fact_gives_two_key_sentences():
    keys = fk.Fact("町内会", "夏祭りの日", "8月3日").keys()
    assert keys == [("答え", "町内会の夏祭りの日"), ("主語", "夏祭りの日が「8月3日」であるもの")]


def test_an_index_written_as_digits_is_read():
    got = fk.parse('[{"i": "0", "facts": [{"s": "x", "r": "y", "a": "z"}]}]', 1)
    assert got == [[fk.Fact("x", "y", "z")]]


def test_a_reply_none_of_whose_items_matches_a_record_is_a_failure():
    """どの番号にも当たらない返りを「事実 0 個」として印を付けると、その束は二度と抜かれない。"""
    assert fk.parse('[{"index": 0, "facts": []}]', 2) is None
    # 空の配列は「どの記録からも抜くものが無い」（頼みが空にしてよいと言っている）。
    assert fk.parse("[]", 2) == [[], []]
    # 事実 0 個でも、番号が当たっていれば読めた返り。
    assert fk.parse('[{"i": 0, "facts": []}]', 1) == [[]]


def test_an_empty_subject_gives_a_sentence_of_the_relation_alone():
    assert fk.Fact("", "夏祭りの日", "8月3日").keys()[0] == ("答え", "夏祭りの日")


def test_a_boolean_index_is_not_a_number():
    assert fk.parse('[{"i": true, "facts": []}, {"i": 0, "facts": []}]', 2) == [[], None]


def test_a_bracket_in_a_preface_does_not_break_the_reading():
    raw = '[注意] 以下が結果です。\n[{"i": 0, "facts": [{"s": "x", "r": "y", "a": "z"}]}]'
    assert fk.parse(raw, 1) == [[fk.Fact("x", "y", "z")]]
