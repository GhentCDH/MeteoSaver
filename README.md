# MeteoSaver v1.1

Here we present MeteoSaver v1.1, a machine-learning based software for the transcription of historical weather data.

## Directory structure

Below is the structure for this project.

```
├── README.md                                               <- This includes general information on the project
|                                                              and introduction of project structure and modules
|
├── OCR_HTR_models                                          <- Trained and serialized models, model predictions, or model summaries.
|                                                         
├── data
│   ├── 00_post1960_DRC_hydroclimate_datasheet_images       <- Sample images of climate data sheets from the INERA
|                                                              Yangambi archives, DRC, arranged in folders (representing station numbers)
|   ├── 01_metadata_INERA_stations                          <- Metadata for the INERA meteorological stations
│
├── docs                                                    <- Flow charts and keys for MeteoSaver included in Muheki et al. 2025.
|
├── results                                                 <- All results obtained from exceution of MeteoSaver v1.0 on the sample ten datasheets
|   |                                                          included in this repository as a Minimal Working Example                                                                                         
│   ├──01_pre_QA_QC_transcribed_hydroclimate_data           <- Original automatic transcription using MeteoSaver before QA/QC checks
|   |                                       
│   ├── 02_post_QA_QC_transcribed_hydroclimate_data         <- Transcription after using MeteoSaver QA/QC checks
|   |                                      
│   ├── 03_validation_transcibed_data                       <- Validation of the transcription using MeteoSaver (in comparison to manual transcription)
│   │
│   ├── 04_final_refined_daily_hydroclimate_data            <- Final refined (formatted) trancribed data, in .xlsx and .tsv (SEF) ready for upload
|   │
│   ├── 05_transient_transcription_output                   <- Transient outputs (during processing)
|
│   ├── 06_manually_transcribed_hydroclimate_data           <- Manually transcribed data (For validation purposes)
|
|
|                                       
├── src                                                     <- Modules (2-6): Transcribing code/scripts for MeteoSaver v1.0
│   ├── main.py                                             <- Main script to run all the modules 1-6 of MeteoSaver (scripts)
|   |                                                          i.e. in order (i) configuration, (iI) image-preprocessing module, (iii) table and cell
|   |                                                          detection model, (iv) transcription, (v) quality assessment and control,
|   |                                                          and (vi) data formatting and upload
│   │
│   ├── tui.py                                              <- Dashboard to set the main configuration settings and run the modules from the terminal
│   ├── review_page.py                                      <- Builds the review page of a sheet: crops, text as read, value after QA/QC (column by column)
│   │
│   ├── meteosaver_app                                      <- Scripts of the dashboard (layout, colours, logo, reading/writing of the configuration file)
│   │
│   ├── image_discovery.py                                  <- Script to search the image directory, and all the folders within it, for the images of the data sheets
│   │
│   ├── image_preprocessing_module.py                       <- Script to carry out image preprocessing of the original scans
|   |                                                          of climate data records
│   │
│   ├── table_and_cell_detection_model.py                   <- Script to detect the table and cells from the already
|   |                                                          pre-processed images
│   │
│   ├── transcription.py                                    <- Script to detect the text within the detected cells using
|   |                                                          an Optical Character Recognition (OCR) or Handwritten Text
|   |                                                          Recognition (HTR) model of your choice.
|   │
│   ├── quality_assessment_and_quality_control.py           <- Script to perform QA/QC checks on the original automatically transcribed data
|   |                                       
│   ├── validation.py                                       <- Script to generates a visual comparison of daily maximum, minimum,
|   |                                                          and average temperatures between manually transcribed data and
|   |                                                          QA/QC checked transcribed data for a specific station
│   └── data_formatting_and_upload.py                       <- Script to select the confirmed data (from the QA/QC) and convert it both an excel file 
|                                                              and to the Station Exchange Format, as well plot timeseries per station
|   
│
├── Dockerfile                                              <- Docker install routine for a virtual environment
|
├──LICENSE                                                  <- License
|
├── configuration.ini                                       <- Module 1: Configuration. User-defined settings to ensure smooth running of MeteoSaver
|
├── pyproject.toml                                          <- The requirements for reproducing the analysis environment with uv
|
├── uv.lock                                                 <- The exact versions of all packages in the analysis environment. Generated with `uv lock`
|
├── job_script.sh                                           <- Job script for HPC infrastructure users to run the software
|
└── setup.py                                                <- Make this project pip installable with `pip install -e` 

```

## Setup

The Python environment is managed with [uv](https://docs.astral.sh/uv/) ([see here for uv install instructions](https://docs.astral.sh/uv/getting-started/installation/)). An isolated Docker environment is provided as well.

### uv

To create an environment which is consistent, run the following in the main project directory. This installs Python 3.12 and all packages, in the exact versions listed in `uv.lock`, into the `.venv` folder.

```bash
uv sync
```

Run the software in this environment using:

```bash
uv run python src/main.py
```

To add or update a package, use `uv add <package>` or `uv lock --upgrade-package <package>`. This updates both `pyproject.toml` and `uv.lock`.

> [!NOTE]
> Tesseract-OCR itself is not a Python package and has to be installed separately if you select it as OCR/HTR model (see the [configuration module](configuration.ini)).
> GLM-OCR is installed with the environment (torch and transformers). A GPU is strongly recommended for it; on Linux, uv installs the CUDA version of torch.

### Docker

> [!WARNING]
> A Docker-based setup is recommended for users who want a more isolated and reproducible execution environment. Local Conda or virtual-environment installations are also supported, but containerization may reduce dependency and system-configuration issues across platforms.

The Docker setup has not been moved to uv yet: it builds from the former `environment.yml` and does not include GLM-OCR.

The dockerfile included provides a Conda environment ([see here for docker install instructions](https://docs.docker.com/engine/install/)).
You can build this docker image using the below command. This will download all required
python components and packages, while safeguarding (sandboxing) your system
from `pip` based [security issues](https://www.bleepingcomputer.com/news/security/pypi-suspends-new-user-registration-to-block-malware-campaign/). Once build locally no further downloads 
will be required.

```
# In the main project directory run
docker build -f Dockerfile -t meteosaver_env .
```

To spin up a docker image using:

```
docker run -it -v /local_data:/docker_data_dir meteosaver_env
```

## Modules
The figure below represents the modules in MeteoSaver v1.0

  ![Schematic representation of the modules in MeteoSaver v1.0](https://github.com/VUB-HYDR/MeteoSaver/blob/6a5238af498088d58173940e04fe8e5cf66567be/docs/Schematic%20representation%20of%20the%20modules%20in%20MeteoSaver%20v1.0.png)


## How to run MeteoSaver v1.0 
After setting up the python environment using uv (see [Setup](#setup)) or Docker setup available on this repository, input the following settings in the [configuration module](https://github.com/VUB-HYDR/MeteoSaver/blob/b8138fa5a23f4ce40603cae8defd82d10734fdbd/configuration.ini) specific to your case study (sheets) before running (see table below):
1. General: Here, you specify the environment in which the scripts will run, i.e. ```local``` (Sequential processing on a personal computer) or ```hpc``` (Parallel processing using multiple processors, suitable for High Performance Computing (HPC) environments). This is set to ```local``` by default
2. Directories. Here, you specify the directories for the following: (i) all historical weather data sheet images in folders per station, (ii) pre-QA/QC transcribed data, (iii) post-QA/QC transcribed data, (iv) the final refined daily hydroclimate data (after all quality checks), (v) transient transcription output during processing, (vi) manually transcribed data (used for validation), (vii) alidation results comparing manually transcribed and the MeteoSaver transcribed data, and (viii) all the stations metadata.
3. Table and Cell Detection: User specifications for table and cell detection.
4. Transcription: User specifications related to the Optical Character Recognition/Handwritten Text Recognition (OCR/HTR).
5. QA/QC: Here, you specify parts of the transcribed table on which to perform QA/QC checks.
6. Data Formatting: Here, you specify the location of the date information in the tables, used for formatting the transcribed data to time series in .xlsx and .tsv (Station Exchange Format).

After inputting the configuration settings specific to your case study (see Table below), you can then run the [main.py](https://github.com/VUB-HYDR/MeteoSaver/blob/7aeab0f526b44056c062407df7cfe467e20a67d8/src/main.py) script if using ```local```, or [job_script](https://github.com/VUB-HYDR/MeteoSaver/blob/aefb8d8068762c5a7715be24dfee835363673ae7/job_script.sh) in case of ```hpc``` architecture, which runs all the modules 1-6 of MeteoSaver i.e. in order (i) configuration, (iI) image-preprocessing module, (iii) table and cell detection model, (iv) transcription, (v) quality assessment and control, and (vi) data formatting and upload, and return results in the specified directories. 

### Dashboard
Instead of editing the configuration file by hand and running [main.py](src/main.py), you can use the dashboard in the terminal:

```bash
uv run python src/tui.py
```

The dashboard shows the main settings of the [configuration module](configuration.ini) (image and results directories, cell detection engine, OCR/HTR model, and the temperature thresholds of the QA/QC), lets you select which modules to run (transcription, clean-up, quality control, data formatting, validation) and shows their progress and the log of the run. The settings are saved to the configuration file when a run starts; all other settings are still edited in the configuration file itself. A module that is switched off uses the files of an earlier run that are in the results directories. Use `--config` to work with another configuration file.

The selection of modules is also available without the dashboard, e.g. `uv run python src/main.py --stages qaqc formatting validation`.

#### Cell detection engine
MeteoSaver has two engines for the detection of the table and its cells (`cell_detection_engine` in the configuration file, or the CELL DETECTION panel of the dashboard):
- **opencv** (the default): detects the handwritten text in the table with OpenCV ([table_and_cell_detection_model.py](src/table_and_cell_detection_model.py)). The detected text is then grouped into rows and columns (`row_grouping`).
- **taulu**: detects the ruled grid of the table with [taulu](https://github.com/GhentCDH/taulu) ([taulu_cell_detection.py](src/taulu_cell_detection.py)), starting from an annotated image of the header. Every cell comes with its row and column, so it is placed in the transcribed table as it is; `row_grouping` is not used. The header is annotated once, with the ANNOTATE HEADER button of the dashboard; see [taulu/README.md](taulu/README.md) for the annotation and the settings.

#### Clean-up and quality control
The QA/QC module has two switches. **Clean-up** formats the transcribed text (digits to decimals, trimming of values that are too long) and checks single values (a daily temperature in the 80s read as the 30s, values outside the minimum and maximum temperature, which are marked red). It recalculates nothing and confirms nothing. **Quality control** also does that, and adds the cross-checks that recalculate values and confirm them (marked green): multi-day totals and averages, the daily mean and amplitude, and the vapour pressure, humidity and wet-bulb checks. Quality control includes the clean-up, so with both switched on the second pass replaces the output of the first; `--stages cleanup` and `--stages qaqc` select them without the dashboard.

Data formatting and validation use the values that quality control confirmed. For a sheet that only had the clean-up (the workbook records this), they use every number as it is.

#### Review page
The dashboard has one more switch, **REVIEW PAGE (BROWSER)**, off by default. When it is on, a page is written for each sheet (to `09_review_pages/<station>/<image file>/index.html` next to the other results, or to `review_dir` of the configuration file) and the first one opens in the browser as soon as it is ready. The page shows the table column by column, with the crop of each cell that GLM-OCR read, the text as it was read, the value after QA/QC and, for the temperature columns, the manual transcription if there is one in `manually_transcribed_data_dir`. Green and red mark agreement with the manual value; amber marks a reading that QA/QC changed. Use the left and right arrow keys to change column, and "differences only" to hide the cells that are fine.

The page needs the GLM-OCR model (it shows the text as GLM-OCR read it). Without the dashboard: `uv run python src/main.py --stages transcription qaqc review`; the module is [review_page.py](src/review_page.py).

### Minimal Working Example (MWE)
You can run the entire script in this repository as a Minimal Working Example (MWE) without modifying any configuration settings. Simply set up the Python environment on your personal computer, i.e. ```local```, and execute the script using [main.py](https://github.com/VUB-HYDR/MeteoSaver/blob/7aeab0f526b44056c062407df7cfe467e20a67d8/src/main.py) or run the MWE provided on this repository ([here](https://github.com/VUB-HYDR/MeteoSaver/tree/81c420360c39258ea67087764dee15d8ceb634cb/manual_and_minimal_working_example)) in Jupyter Notebooks.


### Configuration user-settings
The table below describes all the configuration user-settings.
![Configuration_user_settings](https://github.com/VUB-HYDR/MeteoSaver/blob/80195a7a08ed4c0435c55c0a3e4cba907d9fe0e9/docs/Configuration%20user%20settings.png)




## Authors
Derrick Muheki

Koen Hufkens

Bas Vercruysse

Krishna Kumar Thirukokaranam Chandrasekar

Wim Thiery


## Acknowledgements

This template was inspired by the [python_proj_template](https://github.com/pepaaran/python_proj_template).
