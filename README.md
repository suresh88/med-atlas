# med-atlas

## Running the GPT-powered ICD-10 lookup

The ICD-10 mapper now wraps the OpenAI Responses API. Set your `OPENAI_API_KEY`
environment variable and then run the CLI with either explicit diseases or a
drug name pulled from the RxNav workbook:

```bash
# Look up codes for a drug's associated diseases (example uses acetaminophen)
python -m src.icd10_mapper --drug-name "Acetaminophen"

# Or provide diseases directly
python -m src.icd10_mapper --disease "Hypertension" --disease "Type 2 Diabetes"
```

The command prints a JSON list of disease/code mappings. Use `--excel-path` to
point at a different workbook and `--model` to change the GPT model name.
