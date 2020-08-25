# SODA
This repository is the imprimentation of "SODA: Story Oriented Dense Video Captioning Evaluation Flamework" published at ECCV 2020.
SODA measures the performance of video story description systems.

## Requirements
python 3.6+ (developed with 3.7)
* Numpy
* tqdm
* [pycocoevalcap (Python3 version)](https://github.com/salaniz/pycocoevalcap)
* BERTScore (optional)

## Usage
You can run SODA by specifying the path of system output and that of ground truth.
Both files should be the json format for ActivityNet Captions.
```bash
python soda.py -s path/to/submission.json -r path/to/ground_truth.json 
```

You can try other sentence evaluation metrics, e.g. CIDEr and BERTScore, with `-m` option.
```bash
python soda.py -s path/to/submission.json -m CIDEr
```

## Sample input file
Please use the same format as [ActivityNet Challenge](http://activity-net.org/index.html)
```
{
  version: "VERSION 1.0",
  results: {
    "sample_id" : [
        {
        sentence: "This is a sample caption.",
        timestamp: [1.23, 4.56]
        },
        {
        sentence: "This is a sample caption 2.",
        timestamp: [7.89, 19.87]
        }
    ]
  }
  external_data: {
    used: False,
    }
}
```

## Reference
```
@inproceedings{Fujita2020soda,
  title={SODA: Story Oriented Dense Video Captioning Evaluation Flamework},
  author={Soichiro Fujita and Tsutomu Hirao and Hidetaka Kamigaito and Manabu Okumura and Masaaki Nagata},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  month={August},
  year={2020},
}
```

## LISENCE
NTT Lisence

According to the lisence, it is not allowed to create pull requests.
Please feel free to send issues.
