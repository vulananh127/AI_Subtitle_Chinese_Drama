"""Bảng thông tin ngôn ngữ (mã ngắn như trong DRAMA_LANGUAGES: zh/ko/ja + vi/en)."""

LANGS = {
    "zh": dict(name="Chinese", full="Chinese (Simplified)", name_zh="中文",
               nllb="zho_Hans", whisper="chinese", qwen="Chinese", bleu_tok="zh"),
    "ko": dict(name="Korean", full="Korean", name_zh="韩语",
               nllb="kor_Hang", whisper="korean", qwen="Korean", bleu_tok="ko-mecab"),
    "ja": dict(name="Japanese", full="Japanese", name_zh="日语",
               nllb="jpn_Jpan", whisper="japanese", qwen="Japanese", bleu_tok="ja-mecab"),
    "vi": dict(name="Vietnamese", full="Vietnamese", name_zh="越南语",
               nllb="vie_Latn", whisper="vietnamese", qwen="Vietnamese", bleu_tok="13a"),
    "en": dict(name="English", full="English", name_zh="英语",
               nllb="eng_Latn", whisper="english", qwen="English", bleu_tok="13a"),
}


def lang_info(code: str) -> dict:
    if code not in LANGS:
        raise KeyError(f"Chưa có thông tin ngôn ngữ '{code}' trong langs.LANGS (có: {sorted(LANGS)})")
    return LANGS[code]


def prompt_fields(src: str, src_code: str, tgt_code: str) -> dict:
    """Các trường dùng được trong prompt template của MT: {src} {src_name} {tgt_name} {src_full} {tgt_full}
    {src_code} {tgt_code} {src_name_zh} {tgt_name_zh}"""
    s, t = lang_info(src_code), lang_info(tgt_code)
    return dict(src=src, src_name=s["name"], tgt_name=t["name"], src_full=s["full"], tgt_full=t["full"],
                src_code=src_code, tgt_code=tgt_code, src_name_zh=s["name_zh"], tgt_name_zh=t["name_zh"])
