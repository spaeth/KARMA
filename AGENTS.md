# Developer Workflow

This project uses `flake8` for linting and `pytest` for tests. Please run both before committing any changes.

## Linting
1. Install dependencies if needed:
   ```bash
   pip install flake8
   ```
2. Run linting from the repository root:
   ```bash
   flake8 .
   ```

## Testing
1. Install pytest if needed:
   ```bash
   pip install pytest
   ```
2. Execute tests:
   ```bash
   pytest -q
   ```

## Environment Variables
The pipeline expects the environment variable `OPENAI_KEY` to contain your OpenAI API key. This key is passed to `karma_pipeline.py` when initializing the `OpenAI` client.
Example:
```bash
export OPENAI_KEY=your-key-here
python karma_pipeline.py
```

## Domain-Specific Guidance: German Tax Law
The pipeline can be adapted to automatically process German tax statutes from **gesetze-im-internet.de**. Use a script with `requests` or `urllib` to download the XML versions of the laws (e.g. `https://www.gesetze-im-internet.de/estg/xml.zip`). Unzip the archive and parse the XML into paragraphs or subsections and feed them into the agents.

After each major stage (ingestion, reading, extraction, alignment, evaluation) run an end-to-end test with the text of **§1 EStG** to ensure the pipeline behaves correctly. The agents may call the OpenAI API and therefore require `OPENAI_KEY` to be set.

### Entity Classes
Extract entities as nodes in the knowledge graph. The following classes are recommended (in hierarchical order):
- **Gesetz**
- **Rechtsnorm / Paragraph / Artikel**
- **Tatbestand**
- **Rechtsfolge**
- **Begriff / Legaldefinition**
- **Steuerart**
  - Einkommensteuer (ESt)
  - Körperschaftsteuer (KSt)
  - Umsatzsteuer (USt)
  - Erbschaft- und Schenkungsteuer (ErbSt)
  - Gewerbesteuer (GewSt)
  - Grunderwerbsteuer (GrESt)
  - Solidaritätszuschlag (SolZ)
  - sonstige Bundes- und Landessteuern (EnergieSt, StromSt ...)
- **Steuersubjekt / Steuerpflichtiger**
  - Natürliche Person
  - Juristische Person
  - Personengesellschaft / Mitunternehmerschaft
- **Steuerobjekt**
- **Bemessungsgrundlage**
- **Steuersatz / Steuertarif**
- **Freibetrag**
- **Steuerbefreiung**
- **Steuerklasse**
- **Erhebungsform / Erhebungsverfahren**
- **Veranlagungszeitraum / Stichtag**
- **Frist / Termin**
- **Steuererklärung / Anmeldung**
- **Steuerbescheid**
- **Steuerbetrag / Zahlung**
- **Steuermessbetrag**
- **Behörde / Finanzamt / Gemeinde**
- **Abgabenordnung (AO) – Verfahrensgesetz**

### Relationship Classes
Model the following relationships as edges:
- **Gesetz enthält Norm**
- **Norm verweist auf Norm**
- **Norm hat Tatbestand**
- **Norm hat Rechtsfolge**
- **Norm definiert Begriff**
- **Tatbestand führt zu Rechtsfolge**
- **Steuerart hat Steuerschuldner**
- **Steuerart hat Steuerobjekt**
- **Steuerart hat Bemessungsgrundlage**
- **Steuerart hat Steuersatz / Tarif**
- **Steuerart gewährt Freibetrag**
- **Steuerart hat Steuerbefreiung**
- **Steuer wird erhoben durch Erhebungsform**
- **Erhebungsform hat Frist / Termin**
- **Steuerart wird verwaltet von Behörde**
- **Steuerpflichtiger ist steuerpflichtig für Steuerart**
- **Steuerpflichtiger hat Steuerklasse**
- **Steuerpflichtiger schuldet Steuerbetrag**
- **Steuerpflichtiger legt Einspruch ein gegen Bescheid**
- **Steuerbescheid bezieht sich auf Veranlagungszeitraum**
- **Steuerbetrag wird angerechnet auf Steuerbetrag**
- **Steuerbescheid ergibt Zahlung**

These entities and relationships cover the core concepts required for a comprehensive knowledge graph of German tax law.
