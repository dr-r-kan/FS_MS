This is a repository for applying microstate methods to clinical EEG in seizures (ES and FS)

There are many EEGs in two main folders, each with an excel spreadsheet with metadata.

We need to combine the data into one sensible csv.

We have many edf files, but ech might contain one or more event which can be ictal or interictal (II)

The data we want is the age, gender, some sort of file label, patient type, seizure type (for the ictal events), hypo/hyper (for the ictal events), then ict or ii.

Int he metaeventmap the filename stem is the "init" column. In the FS_DD_ES_Database, the FileName column is the same idea (though it has an N on it to indicate where the number goes).

The files have different ways of expressing the multiple events. In MetaEventMap it is one row per event.
In FS_DD_ES it is a row per patient, with the columns:
II start 1	II end 1	II start 2	II end 2	II start 3	II end 3	II start 4	II end 4
for initterictal segments
and
Event 1 EDF	Event 1 date	Event 1 onset HHMMSS	Event 1 offset HHMMSS	ES/FS Event 1	Event 1 classification	Event 1 score (FS)
for each ictal event.

FS_DD_ES uses "category; FS, DD, ES" for patientType, also, it has two different sheets:
Database_demographics 9for demographics) and Database_events for the events on the EEG.

Synonyms:
NEAD: FS
Epileptic seizure: ES

DD: dual diagnosis ES and FS

## HEPPy

HEPPy is tracked as a git submodule in `external/HEPPy`.

```powershell
git submodule update --init --recursive
pip install -r requirements-heppy.txt
```

Fast external ECG review, using the same HEPPy review CSV:

```powershell
python fast_ecg_review.py
python fast_ecg_review.py --qrs 1
python fast_ecg_review.py --auto-fit 1
```

The first command is an instant cached-PNG reviewer: `y` yes, `n` no, `f` flip, arrows move. The second opens a fast QRS overlay for one row/base name: drag-select odd beats, then `r` refit, `d` delete, `a` accept. The third dry-runs semi-automatic QRS correction at 5%; flagged beats are aligned to the average QRS by template correlation and dropped if still poor. Add `--auto-save --auto-plot` to write corrected peaks and QC PNGs, and `--auto-mark-viable` only after you are happy with the plots.

Once ECG review is complete, run HEPPy preprocessing without opening the Streamlit app:

```powershell
python run_heppy_preprocessing.py
```

By default this processes only `viable=yes` rows and refuses to run while review rows are still blank. Use `--allow-pending` to process current yes rows only, or `--process-pending` if the ECG has been fixed outside the reviewer and blanks should be treated as viable.

## VG Microstates

VG-microstates is tracked as a git submodule in `external/VG-microstates`.

After HEPPy has written `raw_fif/*_pp_raw.fif`, build the condition manifest and run the hierarchy:

```powershell
python vg_microstates_pipeline.py --format mat --run-matlab
```

This crops HEPPy patient snip FIFs into `II`, `preict`, and `postict` records, ignores ictal, converts them for MATLAB, then calls VG global/group/condition/participant-condition fitting plus TANOVA.

Scalar GFP/coverage/dwell-time summaries and the hypo/hyper model screen:

```powershell
python analyse_vg_microstate_outputs.py --pipeline-dir D:\EEGData\FS_microstates_study\vg_microstates
```
