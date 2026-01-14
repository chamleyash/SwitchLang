from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import nltk
import os
from nltk import sent_tokenize
from langdetect import detect
import google.generativeai as genai

# Correct path to the model files
cache_dir = "C:/Users/YASH/.cache/huggingface/hub/models--facebook--mbart-large-50-many-to-many-mmt"

# Load the tokenizer and model from the local cache
tokenizer = MBart50TokenizerFast.from_pretrained(cache_dir)
model = MBartForConditionalGeneration.from_pretrained(cache_dir)

def map_lang_code(detected_lang):
    lang_mapping = {
        "af": "af_ZA", "ar": "ar_AR", "az": "az_AZ", "bn": "bn_IN", "my": "my_MM",
        "zh": "zh_CN", "hr": "hr_HR", "cs": "cs_CZ", "nl": "nl_XX", "en": "en_XX",
        "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gl": "gl_ES", "ka": "ka_GE",
        "de": "de_DE", "gu": "gu_IN", "he": "he_IL", "hi": "hi_IN", "id": "id_ID",
        "it": "it_IT", "ja": "ja_XX", "kk": "kk_KZ", "km": "km_KH", "ko": "ko_KR",
        "lv": "lv_LV", "lt": "lt_LT", "mk": "mk_MK", "ml": "ml_IN", "mr": "mr_IN",
        "mn": "mn_MN", "ne": "ne_NP", "fa": "fa_IR", "pl": "pl_PL", "ps": "ps_AF",
        "pt": "pt_XX", "ro": "ro_RO", "ru": "ru_RU", "si": "si_LK", "sl": "sl_SI",
        "es": "es_XX", "sw": "sw_KE", "sv": "sv_SE", "ta": "ta_IN", "te": "te_IN",
        "th": "th_TH", "tr": "tr_TR", "uk": "uk_UA", "ur": "ur_PK", "vi": "vi_VN",
        "xh": "xh_ZA"
    }
    return lang_mapping.get(detected_lang, "en_XX")

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=512
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

genai.configure(api_key="")

def gemini_translate(text, target_lang):
    lang_map = {
         "af_ZA": "Afrikaans", "ar_AR": "Arabic", "az_AZ": "Azerbaijani", "bn_IN": "Bengali",
        "my_MM": "Burmese", "zh_CN": "Chinese", "hr_HR": "Croatian", "cs_CZ": "Czech",
        "nl_XX": "Dutch", "en_XX": "English", "et_EE": "Estonian", "fi_FI": "Finnish",
        "fr_XX": "French", "gl_ES": "Galician", "ka_GE": "Georgian", "de_DE": "German",
        "gu_IN": "Gujarati", "he_IL": "Hebrew", "hi_IN": "Hindi", "id_ID": "Indonesian",
        "it_IT": "Italian", "ja_XX": "Japanese", "kk_KZ": "Kazakh", "km_KH": "Khmer",
        "ko_KR": "Korean", "lv_LV": "Latvian", "lt_LT": "Lithuanian", "mk_MK": "Macedonian",
        "ml_IN": "Malayalam", "mr_IN": "Marathi", "mn_MN": "Mongolian", "ne_NP": "Nepali",
        "fa_IR": "Persian", "pl_PL": "Polish", "ps_AF": "Pashto", "pt_XX": "Portuguese",
        "ro_RO": "Romanian", "ru_RU": "Russian", "si_LK": "Sinhala", "sl_SI": "Slovenian",
        "es_XX": "Spanish", "sw_KE": "Swahili", "sv_SE": "Swedish", "ta_IN": "Tamil",
        "te_IN": "Telugu", "th_TH": "Thai", "tr_TR": "Turkish", "uk_UA": "Ukrainian",
        "ur_PK": "Urdu", "vi_VN": "Vietnamese", "xh_ZA": "Xhosa"
    }
    lang_name = lang_map.get(target_lang, "English")
    prompt = f"Work like a translator and give me one single answer for this query.\n Translate the following text to {lang_name}:\n\n{text}"
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    response = model_gemini.generate_content(prompt)
    return response.text.strip() if hasattr(response, "text") else str(response).strip()

def translate_code_switched_to_target(text, target_lang, use_gemini_final=False):
    sentences = sent_tokenize(text)
    english_segments = []
    for sentence in sentences:
        try:
            detected_lang = detect(sentence)
        except:
            detected_lang = "en"
        src_lang = map_lang_code(detected_lang)
        english = translate(sentence, src_lang, "en_XX")
        english_segments.append(english)
    combined_english = ' '.join(english_segments)
    if target_lang == "en_XX":
        return combined_english
    final_translation = translate(combined_english, "en_XX", target_lang)
    return gemini_translate(final_translation, target_lang)