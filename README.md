To use this translator, you'd need the mBART model whose link I'll be providing and also a Gemini API key.
mBART model link - https://huggingface.co/facebook/mbart-large-50/tree/main
There is no need to download all the files. You only need to download:
1. congig.json
2. pytorch_model.bin
3. sentence.bpe.model
4. tokenizer_config.json
Store these files in a single folder and copy the path of this folder and store it in SwitchLang -> mbart_model -> tranlator.py -> cached_dir.

To create Gemini API key - https://ai.google.dev/gemini-api/docs/api-key
After creating the API key, in the same translator.py paste it in line 47

To run the projectopen the terminal of the project folder, and run the python command:
python manage.py runserver

NOTE: Install all the dependencies required.
