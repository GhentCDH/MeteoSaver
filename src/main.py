import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import configparser
import traceback
import multiprocessing as mp
from datetime import datetime
import glob
import pandas as pd

# Import all the required modules
from image_preprocessing_module import *
from table_and_cell_detection_model import *
from transcription import *
from quality_assessment_and_quality_control import *
from data_formatting_and_upload import *
from validation import *
from review_page import build_review_page
from image_discovery import find_sheets

# Logger Setup to inform users of progress through the multiple modules/stages
from logger_setup import setup_logger
log_dir = 'logs'
logger = setup_logger(log_dir)

# Module 1: Configuration
# Load settings from user configurations. See configuration.ini file in this repository
config = configparser.ConfigParser()
# Adjust path to config.ini (since it's placed in the root directory)
config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configuration.ini')
config_file_path = os.environ.get('METEOSAVER_CONFIG', config_file_path) # Optionally, another configuration file can be selected with this environment variable
config.read(config_file_path)

# Get run_mode and number of processors from configuration file
run_mode = config['General']['run_mode']
num_processors = int(config['General'].get('num_processors', 1))

# Set up directories from the configuration file
full_datadir = config['Directories']['full_datadir'] # Directory to all the images/scans of the hydroclimatic data sheets
pre_QA_QC_transcribed_hydroclimate_data_dir = config['Directories']['pre_QA_QC_transcribed_hydroclimate_data_dir'] # Directory where pre-QA/QC transcribed data is stored
post_QA_QC_transcribed_hydroclimate_data_dir = config['Directories']['post_QA_QC_transcribed_hydroclimate_data_dir'] # Directory where post-QA/QC transcribed data is stored
final_refined_daily_hydroclimate_data_dir = config['Directories']['final_refined_daily_hydroclimate_data_dir'] # Directory for the final refined daily hydroclimate data (after all quality checks)
manually_transcribed_data_dir = config['Directories']['manually_transcribed_data_dir'] # Directory for manually transcribed data (used for validation)
validation_dir = config['Directories']['validation_dir'] # Directory for validation results comparing manually transcribed and the MeteoSaver transcribed data 
transient_transcription_output_dir = config['Directories']['transient_transcription_output_dir'] # Directory to store transient transcription output during processing
metadata_file_path = config['Directories']['metadata_file_path'] # Directory for all the stations metadata
formatted_already_digitized_data_dir = config['Directories']['formatted_already_digitized_data_dir'] # Directory for previously transcribed data to add to our new transcribed database
region_shapefile_path = config['Directories']['region_shapefile_path'] # Directory for shapefile of the study region
result_maps_dir = config['Directories']['result_maps_dir'] # Directory for results overlain on maps
glm_output_dir = config['Directories'].get('glm_output_dir', os.path.join(os.path.dirname(os.path.normpath(pre_QA_QC_transcribed_hydroclimate_data_dir)), '08_GLM_OCR_output')) # Directory for the text as read by GLM-OCR, per sheet

review_dir = config['Directories'].get('review_dir', os.path.join(os.path.dirname(os.path.normpath(pre_QA_QC_transcribed_hydroclimate_data_dir)), '09_review_pages')) # Directory for the review pages (one per sheet, see review_page.py)

# Search full_datadir, and all the folders within it, for the images of the data sheets
sheets_by_station, search_messages = find_sheets(full_datadir)
all_stations = list(sheets_by_station)

# Modules to run. By default all of them; a selection can be given on the command line, e.g. --stages qaqc formatting validation
# (the modules that are left out then use the files that are already in the results directories)
# 'cleanup' is the part of the QA/QC module that does not recalculate anything (formatting of the transcribed text, single-value checks);
# 'qaqc' is the quality control with the cross-checks that recalculate and confirm values (it includes the clean-up)
ALL_STAGES = ['transcription', 'cleanup', 'qaqc', 'formatting', 'validation']
argument_parser = argparse.ArgumentParser()
# The review page of each sheet (see review_page.py) is not run by default: it has to be asked for with --stages (the dashboard has a toggle for it)
argument_parser.add_argument('--stages', nargs='+', choices=ALL_STAGES + ['review'], default=ALL_STAGES)
stages_to_run = argument_parser.parse_known_args()[0].stages

def report_error(message):
    # Reports an error with its traceback, marked for the dashboard, without stopping the other sheets and stations
    print(f"[ ERROR ] {message}", flush=True)
    for line in traceback.format_exc().rstrip().splitlines():
        print(f"[ ERROR ] | {line}", flush=True)

def announce_stage(stage, event):
    # Marks the start and the end of a module in the output, for the dashboard (src/tui.py) to follow the progress
    print(f"[stage:{stage}] {event}", flush=True)

def detect_table_and_cells(month_data, month_filename, station):
    # Modules 2 and 3: image pre-processing, and table and cell detection of one sheet
    logger.info("Step 1: Preprocessing image")
    image_in_grayscale, binarized_image, original_image = image_preprocessing(month_data,
        binarization_block_size = int(config['TableAndCellDetection'].get('binarization_block_size', 51)),
        binarization_offset = int(config['TableAndCellDetection'].get('binarization_offset', 6)))

    logger.info("Step 2: Detecting table and cells")
    return table_and_cell_detection(image_in_grayscale, binarized_image, original_image, station, month_filename, transient_transcription_output_dir,
        clip_up = int(config['TableAndCellDetection']['clip_up']), 
        clip_down = int(config['TableAndCellDetection']['clip_down']),
        clip_left = int(config['TableAndCellDetection']['clip_left']),
        clip_right = int(config['TableAndCellDetection']['clip_right']),
        max_table_width = int(config['TableAndCellDetection']['max_table_width']),
        max_table_height = int(config['TableAndCellDetection']['max_table_height']),
        min_cell_width_threshold=int(config['TableAndCellDetection']['min_cell_width_threshold']),
        max_cell_width_threshold=int(config['TableAndCellDetection']['max_cell_width_threshold']),
        min_cell_height_threshold=int(config['TableAndCellDetection']['min_cell_height_threshold']),
        max_cell_height_threshold=int(config['TableAndCellDetection']['max_cell_height_threshold']),
        space_height_threshold=int(config['TableAndCellDetection']['space_height_threshold']), 
        space_width_threshold=int(config['TableAndCellDetection']['space_width_threshold']), 
        max_cell_height_per_box=int(config['TableAndCellDetection']['max_cell_height_per_box']), 
        no_of_rows=int(config['TableAndCellDetection']['no_of_rows']), 
        no_of_columns=int(config['TableAndCellDetection']['no_of_columns']),
        line_kernel_divisor_vertical=int(config['TableAndCellDetection'].get('line_kernel_divisor_vertical', 50)),
        line_kernel_divisor_horizontal=int(config['TableAndCellDetection'].get('line_kernel_divisor_horizontal', 60)),
        line_dilation_iterations_vertical=int(config['TableAndCellDetection'].get('line_dilation_iterations_vertical', 5)),
        line_dilation_iterations_horizontal=int(config['TableAndCellDetection'].get('line_dilation_iterations_horizontal', 5)))

# Function to process each station's data
def process_station(station):
    pre_QA_QC_transcribed_hydroclimate_data_dir_station = os.path.join(pre_QA_QC_transcribed_hydroclimate_data_dir, station)
    post_QA_QC_transcribed_hydroclimate_data_dir_station = os.path.join(post_QA_QC_transcribed_hydroclimate_data_dir, station)
    
    # Ensure directories exist
    os.makedirs(pre_QA_QC_transcribed_hydroclimate_data_dir_station, exist_ok=True)
    os.makedirs(post_QA_QC_transcribed_hydroclimate_data_dir_station, exist_ok=True)
    
    # All the images of standard format sheets that were found for this station
    sf_files = sheets_by_station[station]

    for sheet_number, month_data in enumerate(sf_files, start=1):
        month_filename = os.path.basename(month_data)
        print(f"[progress] station={station} ({all_stations.index(station) + 1}/{len(all_stations)}) sheet={sheet_number}/{len(sf_files)} file={month_filename}", flush=True)

        if not os.path.exists(month_data):
            print(f"[ ERROR ] Image not found: {month_data}", flush=True)
            continue

        # CHECKPOINTING: Skip already-processed months. Incase of an HPC job or local pc run stopping before completion. **Uncomment the lines below incase of large volumes of data and limited RUNTIME on resources.
        # output_file = os.path.join(post_QA_QC_transcribed_hydroclimate_data_dir_station, f"{month_filename}_post_QA_QC.xlsx")
        # if os.path.exists(output_file):
        #     print(f"[SKIP] {month_filename} already processed (based on .xlsx).")
        #     continue

        detected_table_and_cells = None  # filled by the transcription module (or by the review module, if only that runs)

        # Perform Pre-processing, Transcription, QA/QC, and Post-processing
        try:
            logger.info(f"Processing station {station}, file: {month_filename}")
            if 'transcription' in stages_to_run:
                announce_stage('transcription', 'start')
                # Modules 2 and 3: image pre-processing, and table and cell detection
                detected_table_and_cells = detect_table_and_cells(month_data, month_filename, station)
            
                # Module 4: Transcription
                logger.info("Step 3: Transcribing values using OCR/HTR")
                start_time = datetime.now()
                ocr_model = config['Transcription']['ocr_model'] # Selected OCR/HTR model
                # Incase of Tesseract
                # Ensure that the tesseract path is set correctly for your local system
                tesseract_path = config['Transcription']['tesseract_path']
                # Set TESSDATA_PREFIX to the system's tessdata directory (for system-wide language files)
                system_tessdata_dir = config['Transcription']['system_tessdata_dir']
                os.environ["TESSDATA_PREFIX"] = system_tessdata_dir
                transcribed_table = transcription(detected_table_and_cells, ocr_model, tesseract_path, transient_transcription_output_dir, pre_QA_QC_transcribed_hydroclimate_data_dir_station, station, month_filename,
                                                  no_of_rows=int(config['TableAndCellDetection']['no_of_rows']),
                                                  no_of_columns=int(config['TableAndCellDetection']['no_of_columns']),
                                                  no_of_rows_including_headers=int(config['TableAndCellDetection']['no_of_rows_including_headers']),
                                                  row_grouping=config['Transcription'].get('row_grouping', 'kmeans'),
                                                  glm_model_dir=config['Transcription'].get('glm_model_dir'),
                                                  glm_python=config['Transcription'].get('glm_python', ''),
                                                  glm_resize_height=int(config['Transcription'].get('glm_resize_height', 128)),
                                                  glm_batch_size=int(config['Transcription'].get('glm_batch_size', 64)),
                                                  glm_output_dir_station=os.path.join(glm_output_dir, station))
            
                end_time = datetime.now()

                print(f'Duration of transcribing: {end_time - start_time}')
                announce_stage('transcription', 'done')
            else:
                # Use the table that was transcribed in an earlier run
                transcribed_table = os.path.join(pre_QA_QC_transcribed_hydroclimate_data_dir_station, f'{month_filename}_pre_QA_QC.xlsx')
                if ('cleanup' in stages_to_run or 'qaqc' in stages_to_run) and not os.path.exists(transcribed_table):
                    print(f"[ WARN ] No transcribed table found for {month_filename} ({transcribed_table}). Skipping.", flush=True)
                    continue

            # Module 5: Quality assessment and Quality Control, in two parts that can be switched separately:
            #  - clean-up: formatting of the transcribed text and checks of single values; nothing is recalculated or confirmed
            #  - quality control: the cross-checks that recalculate values (totals and averages, daily mean and amplitude, vapour pressure
            #    and humidity) and confirm them. Includes the clean-up, so with both switched on the second pass replaces the output of the first.
            for stage, cross_checks in (('cleanup', False), ('qaqc', True)):
                if stage not in stages_to_run:
                    continue
                announce_stage(stage, 'start')
                logger.info(f"Step 4: Quality assesment and Quality Control of transcribed values ({'with' if cross_checks else 'without'} the cross-checks)")
                qa_qc_checked_data = qa_qc(transcribed_table, station, transient_transcription_output_dir, post_QA_QC_transcribed_hydroclimate_data_dir_station, month_filename,
                                                max_temperature_threshold = float(config['QAQC']['max_temperature_threshold']),
                                                min_temperature_threshold = float(config['QAQC']['min_temperature_threshold']),
                                                decimal_places = int(config['QAQC']['decimal_places']),
                                                uncertainty_margin = float(config['QAQC']['uncertainty_margin']),
                                                header_rows = int(config['QAQC']['header_rows']),
                                                multi_day_totals = config.getboolean('QAQC', 'multi_day_totals'),
                                                multi_day_averages = config.getboolean('QAQC', 'multi_day_averages'),
                                                max_days_for_multi_day_total = int(config['QAQC']['max_days_for_multi_day_total']),
                                                multi_day_totals_rows = list(map(int, config['QAQC']['multi_day_totals_rows'].split(','))),
                                                final_totals_rows = list(map(int, config['QAQC']['final_totals_rows'].split(','))),
                                                excluded_rows = list(map(int, config['QAQC']['excluded_rows'].split(','))),
                                                excluded_columns = list(map(int, config['QAQC']['excluded_columns'].split(','))),
                                                daily_temperature_columns = config['QAQC']['daily_temperature_columns'].split(','),
                                                daily_temperature_columns_and_diurnal_temperature_range = config['QAQC']['daily_temperature_columns_and_diurnal_temperature_range'].split(','),
                                                daily_precipitation_column = config['QAQC']['daily_precipitation_column'].split(','),
                                                dry_and_wet_bulb_temperature_columns = config['QAQC']['dry_and_wet_bulb_temperature_columns'].split(','),
                                                cross_checks = cross_checks)
                announce_stage(stage, 'done')

            if 'review' in stages_to_run:
                announce_stage('review', 'start')
                try:
                    glm_output_xlsx = os.path.join(glm_output_dir, station, f'{month_filename}_GLM_OCR_output.xlsx')
                    post_QA_QC_xlsx = os.path.join(post_QA_QC_transcribed_hydroclimate_data_dir_station, f'{month_filename}_post_QA_QC.xlsx')
                    if not os.path.exists(glm_output_xlsx):
                        print(f"[ WARN ] No review page for {month_filename}: the review page shows the text as read by GLM-OCR, and {glm_output_xlsx} was not found (transcribe with the GLM-OCR model).", flush=True)
                    elif not os.path.exists(post_QA_QC_xlsx):
                        print(f"[ WARN ] No review page for {month_filename}: {post_QA_QC_xlsx} was not found (run the QA/QC module).", flush=True)
                    else:
                        if detected_table_and_cells is None:  # the transcription module did not run: detect the table again, for the images of the cells
                            detected_table_and_cells = detect_table_and_cells(month_data, month_filename, station)
                        page_path = build_review_page(station, month_filename, detected_table_and_cells[3], glm_output_xlsx, post_QA_QC_xlsx,
                                                      os.path.join(review_dir, station, month_filename),
                                                      manual_xlsx=os.path.join(manually_transcribed_data_dir, station, f'{month_filename}_manually_entered_temperatures.xlsx'),
                                                      model_note=f"GLM-OCR ({os.path.basename(os.path.dirname(os.path.normpath(config['Transcription'].get('glm_model_dir', ''))))}), {config['Transcription'].get('row_grouping', 'kmeans')} row grouping")
                        print(f"[review] page={os.path.abspath(page_path)}", flush=True)
                except Exception as e:
                    logger.error(f"Error building the review page of {month_filename}: {e}", exc_info=True)
                    report_error(f"Error building the review page of {month_filename}: {e}")
                announce_stage('review', 'done')

            logger.info(f"Finished processing {month_filename} for station {station}")
        
        except Exception as e:
            logger.error(f"Error processing {month_filename} for station {station}: {e}", exc_info=True)
            report_error(f"Error processing {month_filename}: {e}")
            continue

    # Module 6: Data formatting and Upload
    formatted_data = None
    if 'formatting' in stages_to_run:
        announce_stage('formatting', 'start')
        try:
            final_refined_daily_hydroclimate_data_dir_station = os.path.join(final_refined_daily_hydroclimate_data_dir, station)
            os.makedirs(final_refined_daily_hydroclimate_data_dir_station, exist_ok=True)
    
            logger.info(f"Formatting and integrating QA/QC checked data for station {station}")
            formatted_data = data_formatting(post_QA_QC_transcribed_hydroclimate_data_dir_station, final_refined_daily_hydroclimate_data_dir_station, metadata_file_path, formatted_already_digitized_data_dir, station, 
                            date_column = config['DataFormatting']['date_column'].strip(),
                            header_rows = int(config['QAQC']['header_rows']),
                            multi_day_totals = config.getboolean('QAQC', 'multi_day_totals'),
                            multi_day_averages = config.getboolean('QAQC', 'multi_day_averages'),
                            excluded_rows = list(map(int, config['QAQC']['excluded_rows'].split(','))),
                            additional_excluded_rows = list(map(int, config['QAQC']['additional_excluded_rows'].split(','))),
                            final_totals_rows = list(map(int, config['QAQC']['final_totals_rows'].split(','))),
                            uncertainty_margin = float(config['QAQC']['uncertainty_margin']),
                            max_temperature_threshold = float(config['QAQC']['max_temperature_threshold']),
                            min_temperature_threshold = float(config['QAQC']['min_temperature_threshold']))
        except Exception as e:
            logger.error(f"Error formatting the data of station {station}: {e}", exc_info=True)
            report_error(f"Error formatting the data of station {station}: {e}")
        announce_stage('formatting', 'done')


    # Extra module: Validation
    if 'validation' in stages_to_run:
        announce_stage('validation', 'start')
        try:
            validation_dir_station = os.path.join(validation_dir, station)
            manually_transcribed_data_dir_station = os.path.join(manually_transcribed_data_dir, station)
            # Ensure directories exist
            os.makedirs(validation_dir_station, exist_ok=True)
            os.makedirs(manually_transcribed_data_dir_station, exist_ok=True)

            validate(manually_transcribed_data_dir_station, post_QA_QC_transcribed_hydroclimate_data_dir_station, validation_dir_station, station,
                     date_column = config['DataFormatting']['date_column'].strip(),
                     daily_temperature_columns = config['QAQC']['daily_temperature_columns'].split(','),
                     header_rows = int(config['QAQC']['header_rows']),
                     multi_day_totals = config.getboolean('QAQC', 'multi_day_totals'),
                     multi_day_averages = config.getboolean('QAQC', 'multi_day_averages'),
                     excluded_rows = list(map(int, config['QAQC']['excluded_rows'].split(','))),
                     additional_excluded_rows = list(map(int, config['QAQC']['additional_excluded_rows'].split(','))),
                     final_totals_rows = list(map(int, config['QAQC']['final_totals_rows'].split(','))),
                     uncertainty_margin = float(config['QAQC']['uncertainty_margin']))
            logger.info(f"Validation complete for station {station}")
        except Exception as e:
            logger.error(f"Error validating the data of station {station}: {e}", exc_info=True)
            report_error(f"Error validating the data of station {station}: {e}")
        announce_stage('validation', 'done')

    # Return formatted_data only after validation step completes
    if formatted_data:
        return formatted_data 
    else:
        return None

    
if __name__ == '__main__':
    # Report what the search for images found, and stop if there is nothing to process
    for message in search_messages:
        print(f"[ WARN ] {message}", flush=True)
    if not all_stations:
        print(f"[ ERROR ] No images of data sheets found in {os.path.abspath(full_datadir)} or in the folders within it. Nothing to process.", flush=True)
        raise SystemExit(1)
    print(f"Found {sum(len(sheets) for sheets in sheets_by_station.values())} images of data sheets in {len(all_stations)} stations: {', '.join(all_stations)}", flush=True)

    # Formatted transcribed and qc checked data for all stations. Can be used to calculate for trends
    all_station_trends = []

    if run_mode == 'hpc':
        # HPC Mode: Parallel Processing
        print(f"Running in HPC mode with {num_processors} processors.")
        with mp.Pool(num_processors) as pool:
            all_station_trends = pool.map(process_station, all_stations)
    else:
        # Local Mode: Sequential Processing
        print("Running in Local mode.")
        for station in all_stations:
            result = process_station(station)
            if result:
                all_station_trends.append(result)


