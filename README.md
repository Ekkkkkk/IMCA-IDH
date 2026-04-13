# IDH Core Release

This folder contains a de-identified, shareable core implementation of the modeling pipeline used in the manuscript.

Included modules:
- `idh_core/clustering.py`: Gower distance, PAM clustering, train-fit / test-assign workflow, MDS visualization
- `idh_core/windowing.py`: session-to-window conversion
- `idh_core/models.py`: sequence encoders, meta MLP, WGAN-GP generator / discriminator
- `idh_core/augmentation.py`: WGAN-GP training and positive-class augmentation
- `idh_core/training.py`: metrics, static model training, sequence model training, OOF meta-feature generation
- `idh_core/fusion.py`: logistic-regression meta learner and fusion helpers
- `example_pipeline.py`: a synthetic-data example showing the end-to-end flow

Not included:
- raw EHR data
- hospital-specific preprocessing scripts
- identifiers, database logic, note parsing, or any protected information