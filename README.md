# RealTune
This repository serves as a working space for the RealTune project of the ReTune Hackathon 2023.

## Mission
The aim is to create a common ReTune-wide interface for real-time decoding of neural signals.

## Previous work
Previous work has been done in the following repositories:
 - [py_neuromodulation][pynm_url], which is already being used to calculate features from neural signals by multiple people of the ReTune community.
 - py_neuromodulation also provides some ideas on how to implement [decoding](pynm_decode) and a [real-time stream](pynm_realtime).

## Aims
Some of the specific aims of this project are to:
 1) Define common use cases
 2) Create a flexible interface that can be used across species, tasks, and recording modalities (iEEG, spiking data etc.)
 3) Integrate easily with [py_neuromodulation][pynm_url]
 4) Be **lightweight** (minimal dependencies), **fast** and **easy to use**

## Example implementation
An example on how the API could look like:  
```bash
pip install realtune
```
```python
import time

import numpy as np
import py_neuromodulation as pn
import realtune
import sklearn

sfreq = 1000
model = sklearn.linear_model.LogisticRegression()
processor = pn.DataProcessor(
    sfreq=sfreq,
    settings=settings,
    nm_channels=nm_channels,
)
decoder = realtune.Decoder(model=model)
for i in range(12):
    timestamp = time.time()
    features = processor.process(data=np.random.rand(sfreq, 1))
    label = 0 if i < 5 else 1
    group = i % 2
    decoder.add_features(
        features=features, 
        timestamp=timestamp,
        label=label,
        group=group,
    )
decoder.cross_validate()
decoder.fit_model()
features = processor.process(data=np.random.rand(sfreq, 1))
prediction = decoder.predict(features=features)
```

<!-- Place links and references here -->
[pynm_url]: https://github.com/neuromodulation/py_neuromodulation
[pynm_decode]: https://github.com/neuromodulation/py_neuromodulation/blob/main/py_neuromodulation/nm_decode.py/
[pynm_realtime]: https://github.com/neuromodulation/py_neuromodulation/blob/main/py_neuromodulation/nm_stream_abc.py/