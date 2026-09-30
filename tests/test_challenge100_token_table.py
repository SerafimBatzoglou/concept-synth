import copy
import importlib.util
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest
import yaml


BENCH = Path(__file__).resolve().parents[1] / 'benchmarks/induction'
SPEC = importlib.util.spec_from_file_location('token_table_renderer', BENCH / 'analysis/make_challenge100_leaderboard.py')
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)
REGISTRY = yaml.safe_load((BENCH / 'docs/challenge100_round1_model_registry.yaml').read_text())
USAGE_PATH = BENCH / 'eval/challenge100_output_token_usage.json'
NS = {'s': 'http://www.w3.org/2000/svg'}


def aligned_rows():
    rows = RENDER.render_challenge100(
        registry=REGISTRY,
        eval_path=BENCH / 'eval/induction_challenge64_round1_eval_cache_v1.jsonl',
        holdout_path=BENCH / 'eval/induction_challenge64_round1_holdout_eval_cache_v1.jsonl')
    return RENDER.token_aligned_rows(rows, REGISTRY, RENDER.load_token_usage(USAGE_PATH, REGISTRY))


@pytest.mark.parametrize('missing,expected', [(None, ''), (0, ''), (8, ''), (29, ''), (30, '30 unreported'), (108, '108 unreported')])
def test_material_uncertainty_threshold(missing, expected):
    assert RENDER.token_caveat({'unreported_responses_lower_bound': missing}) == expected


def test_caveats_describe_missing_output_not_known_records_or_missing_input():
    usage = RENDER.load_token_usage(USAGE_PATH, REGISTRY)
    for mid in ['gemini-3.5-flash', 'gemini-3.6-flash', 'gemini-3.7-flash', 'claude-opus-5', 'claude-fable-5-1', 'gpt-6-astra']:
        assert RENDER.token_caveat(usage[mid]) == ''
    assert RENDER.token_caveat(usage['gpt-5.6-sol-xhigh']) == 'C64 unreported'
    assert RENDER.token_caveat(usage['qwen-3.8-max']) == '95 rejected'


def test_output_totals_include_unsuccessful_attempts_and_reasoning_once():
    usage = RENDER.load_token_usage(USAGE_PATH, REGISTRY)
    for path in (BENCH / 'eval').glob('*challenge100_round1_report.json'):
        report = json.loads(path.read_text())
        mid = report['model']
        if mid == 'qwen-3.8-max':
            expected = report['token_usage']['available_token_totals']['output']
        else:
            stats = (report.get('all_physical_response_token_usage') or
                     report.get('known_physical_usage') or report['overall']['token_usage'])
            expected = stats['output']['sum']
            if mid == 'grok-4.7':
                expected += stats['reasoning']['sum']
        assert usage[mid]['output_tokens'] == expected
    # All 129 unsuccessful Sonnet caps contribute 300000 output tokens each.
    sonnet = json.loads((BENCH / 'eval/claude_sonnet_5_5_challenge100_round1_report.json').read_text())
    selected = sonnet['overall']['token_usage']['output']['sum']
    assert usage['claude-sonnet-5-5']['output_tokens'] == selected + 129 * 300000


def test_separate_tables_have_identical_row_positions_and_no_repeated_model_names():
    rows = aligned_rows()
    assert len(rows) == 28
    assert all(re.fullmatch(r'\d+\.\d', row[5]) for row in rows)
    assert rows[0][0] == 'GPT-6 Astra' and rows[0][5:] == ['2.0', '']
    svg = RENDER.render_challenge100_svg(rows)
    assert svg == (BENCH / 'docs/challenge100_leaderboard.svg').read_text()
    root = ET.fromstring(svg)
    left = root.find("s:g[@id='leaderboard']", NS)
    right = root.find("s:g[@id='token-usage']", NS)
    assert int(right.attrib['data-x']) - int(left.attrib['data-width']) == 24
    left_rows = left.findall("s:g[@class='data-row']", NS)
    right_rows = right.findall("s:g[@class='data-row']", NS)
    assert len(left_rows) == len(right_rows) == 28
    for i, (lrow, rrow) in enumerate(zip(left_rows, right_rows)):
        ltext = lrow.findall('s:text', NS)
        rtext = rrow.findall('s:text', NS)
        assert len(ltext) == 5 and len(rtext) == 2
        assert {t.attrib['y'] for t in ltext + rtext} == {str(56 + i * 32 + 21)}
        assert ltext[0].text == rows[i][0]
        assert rtext[0].text == rows[i][5]
        assert (rtext[1].text or '') == rows[i][6]
    names = {m['display_name'] for m in REGISTRY['models']}
    assert not names.intersection(t.text for t in right.iter('{http://www.w3.org/2000/svg}text'))
    markdown = (BENCH / 'docs/leaderboard.md').read_text()
    assert '](challenge100_leaderboard.svg)' in markdown
    assert '<details>\n<summary>Text-only leaderboard and token usage</summary>' in markdown
    assert all('| ' + ' | '.join(row) + ' |' in markdown for row in rows)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'negative'])
def test_bad_usage_data_is_rejected(tmp_path, mutation):
    data = copy.deepcopy(json.loads(USAGE_PATH.read_text()))
    if mutation == 'missing': data['models'].pop()
    elif mutation == 'duplicate': data['models'].append(data['models'][0])
    else: data['models'][0]['output_tokens'] = -1
    path = tmp_path / 'usage.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        RENDER.load_token_usage(path, REGISTRY)


def test_public_usage_data_contains_no_private_paths_or_raw_responses():
    data = USAGE_PATH.read_text()
    for forbidden in ['/Users/', '/tmp/', 'pipeline.sqlite', 'raw_text', 'provider_call_meta', 'OPENAI_API_KEY']:
        assert forbidden not in data
