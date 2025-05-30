from django.shortcuts import render
from langdetect import detect
from mbart_model.translator import translate_code_switched_to_target, map_lang_code

def index(request):
    translation = ''
    if request.method == 'POST':
        text = request.POST.get('text')
        target_lang = request.POST.get('target_lang')
        if text:
            raw_lang = detect(text)
            src_lang = map_lang_code(raw_lang)
            translation = translate_code_switched_to_target(text, target_lang)
    return render(request, 'index.html', {'translation': translation})
